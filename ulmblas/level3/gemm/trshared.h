#ifndef ULMBLAS_LEVEL3_GEMM_TRSHARED_H
#define ULMBLAS_LEVEL3_GEMM_TRSHARED_H 1

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <ulmblas/config/tuned.h>
#include <ulmblas/level3/gemm/pack.h>

//
//  Parts of the frames of trsm and trmm (ulmblas/level3/gemm/frametrsm.h,
//  frametrmm.h) for threads that share the packed blocks of A: the update
//  of the rows outside a diagonal block, C <- C + alpha A_rows X, split
//  into rectangles of whole panels as the threads of gemm (the partition of
//  the gemm frame, GemmFrame::partitionThreads); and the parallel variant.
//
namespace ulmBLAS {

struct TrShared
{
    //  The parallel variants of trsm and trmm on nt threads for an m x m
    //  triangular A and n columns of B (np panels of NR columns):
    //
    //      columns   the columns of B split over the threads (whole panels,
    //                at least minPanels per thread), each thread computes
    //                its columns alone and packs all of A (as OpenBLAS)
    //      shared    the threads share the packed diagonal blocks of A
    //                (each packs some of their panels) and the packed rows
    //                of B of a diagonal block; a diagonal block is split by
    //                columns, the update of the other rows by rows and
    //                columns as the threads of gemm (as BLIS)
    //
    //  columns needs no synchronization, but every thread packs (and reads
    //  from memory) the m^2 / 2 elements of A; shared needs two barriers
    //  per diagonal block (trsm; trmm one).  Measured with dtrsm and dtrmm
    //  n x n, A on the left, on 2 x AMD EPYC 9454 (96 threads) and 2 x
    //  Intel Xeon 8358 (64 threads), GFLOPS of columns (one panel per
    //  thread at least) : shared:
    //
    //                    EPYC 9454                   Xeon 8358
    //      n           trsm        trmm          trsm        trmm
    //      500      693 : 536    724 : 264     524 : 453   694 : 443
    //      1000    1583 : 1425  1614 : 1575   1108 : 938  1278 : 1207
    //      2000    3033 : 2988  3010 : 3226   1719 : 1917 1814 : 2051
    //      4000    3518 : 3587  3505 : 3744   1922 : 2541 1970 : 2776
    //
    //  and on the EPYC 9454 strsm 2000 x 2000 4818 : 3128, strmm 4927 :
    //  3674 (the diagonal blocks, one panel of B per thread, take longer
    //  there), and on the Xeon 8358 ztrsm 4000 x 4000 3121 : 2870 (complex
    //  types: four times the operations per packed element).  The choice:
    //  shared for more than sharedBlocks diagonal blocks (m > sharedBlocks
    //  KB) if there are fewer than sharedPanels panels of B per thread,
    //  otherwise columns with at least minPanels panels per thread; the
    //  parameters per operation and type from tuned() (ulmblas/config/
    //  tuned.h; measured by make tune, by default 4 blocks, AMD Zen 4: 12,
    //  complex types: 4 times that; 16 panels; 1 panel).  (With 4 panels per
    //  thread at least, as before, the EPYC 9454 ran dtrsm 1000 x 1000 on
    //  21 threads: 808 GFLOPS.)  ULMBLAS_TRSM_SHARED, ULMBLAS_TRMM_SHARED =
    //  0|1 (envName) override the variant, ULMBLAS_TRSM_MINP,
    //  ULMBLAS_TRMM_MINP the minimum of panels per thread.
    static std::ptrdiff_t
    envValue(const char *name, std::ptrdiff_t fallback)
    {
	const char *v = std::getenv(name);
	if (!v || v[0] < '0' || v[0] > '9') {
	    return fallback;
	}
	std::ptrdiff_t x = 0;
	for (const char *c = v; *c >= '0' && *c <= '9' && x < 1000000; ++c) {
	    x = 10 * x + std::ptrdiff_t(*c - '0');
	}
	return x;
    }

    //  the variant set by the tests: -1 the choice of useShared, 0
    //  columns, 1 shared
    static std::atomic<std::ptrdiff_t> &
    forced()
    {
	static std::atomic<std::ptrdiff_t> value(-1);
	return value;
    }

    //  shared (true) or columns (false) on nt threads for an m x m A with
    //  diagonal blocks of KB rows and np panels of B; param: the parameters
    //  of the operation (tuned().trsm, tuned().trmm), type: the index of the
    //  type (TunedIndex)
    static bool
    useShared(const char *envName, const Tuned::Tr &param, std::ptrdiff_t type,
              std::ptrdiff_t nt, std::ptrdiff_t m, std::ptrdiff_t KB,
              std::ptrdiff_t np)
    {
	const std::ptrdiff_t f = forced().load(std::memory_order_relaxed);
	if (f >= 0) {
	    return f == 1;
	}
	const std::ptrdiff_t e = envValue(envName, -1);
	if (e >= 0) {
	    return e == 1;
	}
	return m > param.sharedBlocks[type] * KB &&
	       np < param.sharedPanels[type] * nt;
    }

    //  the minimum of panels per thread of columns
    static std::ptrdiff_t
    minPanels(const char *envName, const Tuned::Tr &param, std::ptrdiff_t type)
    {
	const std::ptrdiff_t e = envValue(envName, param.minPanels[type]);
	return e < 1 ? 1 : e;
    }

    //  thread tid's rectangle of the update of rows x nc elements: rows
    //  r0 ... r1-1 (whole panels of MR rows), columns c0 ... c1-1 (whole
    //  panels of NR columns); empty for a thread without one
    template <typename Frame, typename IndexType>
    static void
    rectangle(std::ptrdiff_t tid, std::ptrdiff_t nt, IndexType rows,
              IndexType nc, IndexType &r0, IndexType &r1, IndexType &c0,
              IndexType &c1)
    {
	typedef typename Frame::BackendType Backend;
	constexpr IndexType MR = Backend::MR;
	constexpr IndexType NR = Backend::NR;

	r0 = r1 = c0 = c1 = 0;
	if (rows <= 0 || nc <= 0) {
	    return;
	}
	std::ptrdiff_t jcWays, P;
	Frame::partitionThreads(nt, std::ptrdiff_t(rows), std::ptrdiff_t(nc),
	                        jcWays, P);
	if (tid >= jcWays * P) {
	    return;
	}
	const IndexType jc = IndexType(tid / P), u = IndexType(tid % P);
	const IndexType mp = (rows + MR - 1) / MR, np = (nc + NR - 1) / NR;
	r0 = std::min(rows, u * mp / IndexType(P) * MR);
	r1 = std::min(rows, (u + 1) * mp / IndexType(P) * MR);
	c0 = std::min(nc, jc * np / IndexType(jcWays) * NR);
	c1 = std::min(nc, (jc + 1) * np / IndexType(jcWays) * NR);
    }

    //  C <- C + alpha A X for thread tid: A the rows i0 ... i1-1 of the kb
    //  columns at A (unpacked), X the packed kb x nc block X_ (panels of NR
    //  columns), C the rows i0 ... i1-1 of the nc columns at C; A_ a buffer
    //  of the thread (MCmax kb + MR elements)
    template <typename Frame, typename IndexType, typename TA, typename TC>
    static void
    update(std::ptrdiff_t tid, std::ptrdiff_t nt, IndexType i0, IndexType i1,
           IndexType nc, IndexType kb,
           const typename Frame::BackendType::T &alpha, bool conjA, const TA *A,
           IndexType incRowA, IndexType incColA,
           const typename Frame::BackendType::T *X_, TC *C, IndexType incRowC,
           IndexType incColC, typename Frame::BackendType::T *A_)
    {
	typedef typename Frame::BackendType Backend;
	typedef typename Frame::BlocksType Blocks;
	typedef typename Frame::MacroType Macro;
	typedef typename Backend::T T;

	IndexType r0, r1, c0, c1;
	rectangle<Frame>(tid, nt, i1 - i0, nc, r0, r1, c0, c1);
	if (r1 <= r0 || c1 <= c0) {
	    return;
	}
	IndexType mc;
	for (IndexType i = i0 + r0; i < i0 + r1; i += mc) {
	    mc = Blocks::blockSize(i0 + r1 - i, Blocks::MC, Blocks::MCmax);
	    PackA<Backend>::run(mc, kb, conjA, &A[i * incRowA], incRowA,
	                        incColA, A_);
	    Macro::run(mc, c1 - c0, kb, alpha, A_, &X_[c0 * kb], T(1),
	               &C[i * incRowC + c0 * incColC], incRowC, incColC);
	}
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_TRSHARED_H
