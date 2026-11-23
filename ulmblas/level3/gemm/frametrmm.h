#ifndef ULMBLAS_LEVEL3_GEMM_FRAMETRMM_H
#define ULMBLAS_LEVEL3_GEMM_FRAMETRMM_H 1

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/config/tuned.h>
#include <ulmblas/level3/gemm/frame.h>
#include <ulmblas/level3/gemm/pack.h>
#include <ulmblas/level3/gemm/plan.h>
#include <ulmblas/level3/gemm/threads.h>
#include <ulmblas/level3/gemm/trshared.h>

//
//  The frame algorithm of trmm: B <- alpha A B for a lower (upper)
//  triangular m x m matrix A, B m x n, in place, with the backend, block
//  sizes and macro kernel of a gemm frame Frame (ulmblas/level3/gemm/
//  frame.h).  The other cases (A on the right, op(A) = A^T, A^H) are
//  reduced to these two by the BLAS interface (strides of A and B swapped,
//  see interfaces/blas/C/trmm.cc).
//
//  As OpenBLAS's driver (driver/level3/trmm_L.c), right-looking: for a
//  block of columns of B (nc), the diagonal blocks of A (kb x kb) one after
//  the other, lower from the bottom, upper from the top, so that the rows
//  of B of a block still hold their original values when it is processed.
//  For a diagonal block (rows l ... l+kb-1)
//
//      - the block of B is packed (B_): also the copy that makes the
//        product in place possible, the rows are overwritten afterwards;
//      - the diagonal block of A is packed (A_) and made triangular in the
//        buffer: zeros on the other side of the diagonal, 1 on a unit
//        diagonal (the unstored triangle and a unit diagonal of A may hold
//        anything, e.g. NaN; their values are never used);
//      - B(l:l+kb, :) <- alpha A_ B_ (overwritten) with the micro kernel per
//        panel of MR rows, the k range trimmed to the triangle (lower:
//        columns 0 ... (i+1) MR - 1 for row panel i, upper: i MR ... kb-1),
//        only the MR x MR diagonal tiles compute zeros (as BLIS's trmm macro
//        kernel, bli_trmm_ll_ker_var2, which also trims the k range);
//      - the rows of B below (above) the block are updated with B_:
//        B(rows) <- B(rows) + alpha A(rows, l:l+kb) B_ with the macro kernel
//        of the frame (gemm).
//
//  The diagonal blocks have at most KB rows, as in the frame of trsm
//  (ulmblas/level3/gemm/frametrsm.h).
//
//  On nt threads (GemmThreads::forProblem for the work m^2 n / 2) one of
//  two variants (TrShared, ulmblas/level3/gemm/trshared.h): the columns of
//  B split (as OpenBLAS's gemm_thread_n; every thread computes its columns
//  independently, with its own buffers, no synchronization), or threads
//  that share the packed blocks (runShared, as BLIS).
//
namespace ulmBLAS {

template <typename Frame> struct TrmmFrame
{
    typedef typename Frame::BackendType Backend;
    typedef typename Frame::BlocksType Blocks;
    typedef typename Frame::MacroType Macro;
    typedef typename Backend::T T;

    static constexpr std::ptrdiff_t MR = Backend::MR;
    static constexpr std::ptrdiff_t NR = Backend::NR;

    //  the largest diagonal block for an m x m A: tuned().trmm.kb(m)
    //  (ulmblas/config/tuned.h, by default 256) rounded up to whole panels,
    //  at most KCmax (whole panels) and 2048, at least one panel (test
    //  configurations with KCmax < MR)
    static std::ptrdiff_t
    diagonalBlock(std::ptrdiff_t m)
    {
	constexpr std::ptrdiff_t KC1 = Blocks::KCmax / MR * MR;
	constexpr std::ptrdiff_t KBmax = KC1 < 2048 ? KC1 : 2048 / MR * MR;
	const std::ptrdiff_t kb =
	    std::ptrdiff_t(tuned().trmm.kb[TunedIndex<T>::value](m));
	const std::ptrdiff_t KB = (kb + MR - 1) / MR * MR;
	return KB < MR ? MR : (KB < KBmax ? KB : (KBmax >= MR ? KBmax : MR));
    }

    template <typename IndexType, typename Alpha, typename TA, typename TB>
    static void
    run(bool lower, IndexType m, IndexType n, const Alpha &alpha, bool conjA,
        bool unitDiag, const TA *A, IndexType incRowA, IndexType incColA, TB *B,
        IndexType incRowB, IndexType incColB);

    //  the loops on one thread (alpha != 0)
    template <typename IndexType, typename TA, typename TB>
    static void
    runSerial(bool lower, IndexType m, IndexType n, const T &alpha, bool conjA,
              bool unitDiag, const TA *A, IndexType incRowA, IndexType incColA,
              TB *B, IndexType incRowB, IndexType incColB);

  private:
    //  threads that share the packed blocks of A (see run)
    template <typename IndexType, typename TA, typename TB>
    static void
    runShared(std::ptrdiff_t nt, bool lower, IndexType m, IndexType n,
              const T &alpha, bool conjA, bool unitDiag, const TA *A,
              IndexType incRowA, IndexType incColA, TB *B, IndexType incRowB,
              IndexType incColB);

    //  the panels p0, ..., p1-1 of the packed diagonal block A_ prepared
    template <typename IndexType>
    static void
    prepareDiagonal(bool lower, bool unitDiag, IndexType kb, IndexType p0,
                    IndexType p1, T *A_);

    template <typename IndexType, typename TB>
    static void
    multiplyBlock(bool lower, IndexType kb, IndexType nc, const T &alpha,
                  const T *A_, const T *B_, TB *B, IndexType incRowB,
                  IndexType incColB);
};

//
//  The diagonal tiles of the packed kb x kb block A_ (panels of MR rows,
//  column major, panel p at A_ + p MR kb): zeros on the other side of the
//  diagonal, 1 on a unit diagonal (rows of the tile beyond the block are
//  padding, zeros from the packing).
//
template <typename Frame>
template <typename IndexType>
void
TrmmFrame<Frame>::prepareDiagonal(bool lower, bool unitDiag, IndexType kb,
                                  IndexType p0, IndexType p1, T *A_)
{
    for (IndexType p = p0; p < p1; ++p) {
	const IndexType mr = std::min(IndexType(MR), IndexType(kb - p * MR));
	T *tile = &A_[p * MR * kb + p * MR * MR];
	for (IndexType q = 0; q < mr; ++q) {
	    const IndexType r0 = lower ? 0 : q + 1;
	    const IndexType r1 = lower ? q : mr;
	    for (IndexType r = r0; r < r1; ++r) {
		tile[q * MR + r] = T(0);
	    }
	    if (unitDiag) {
		tile[q * MR + q] = T(1);
	    }
	}
    }
}

//
//  The diagonal block: B <- alpha A_ B_ for the kb x nc block of B, A_ the
//  prepared block, B_ the packed rows of B (panels of NR columns, row major,
//  panel j at B_ + j kb NR).  Row panel i with the k range of the triangle:
//  lower, k = 0 ... min((i+1) MR, kb) - 1, upper, k = i MR ... kb-1 (start
//  i MR MR in the panel of A_, i MR NR in a panel of B_).  beta = 0: the
//  micro kernels do not read B (its rows are saved in B_).
//
template <typename Frame>
template <typename IndexType, typename TB>
void
TrmmFrame<Frame>::multiplyBlock(bool lower, IndexType kb, IndexType nc,
                                const T &alpha, const T *A_, const T *B_, TB *B,
                                IndexType incRowB, IndexType incColB)
{
    const IndexType mp = (kb + MR - 1) / MR;
    const IndexType np = (nc + NR - 1) / NR;

    for (IndexType j = 0; j < np; ++j) {
	const IndexType nr = std::min(IndexType(NR), IndexType(nc - j * NR));
	const T *Bj = &B_[j * kb * NR];

	for (IndexType i = 0; i < mp; ++i) {
	    const IndexType r0 = i * MR;
	    const IndexType mr = std::min(IndexType(MR), kb - r0);
	    const IndexType k0 = lower ? IndexType(0) : r0;
	    const IndexType kk =
	        lower ? std::min(r0 + IndexType(MR), kb) - k0 : kb - k0;
	    const T *Ai = &A_[i * MR * kb + k0 * MR];
	    const T *Bk = &Bj[k0 * NR];
	    TB *Bij = &B[r0 * incRowB + j * NR * incColB];
	    if (mr == MR && nr == NR) {
		Backend::ugemm(kk, alpha, Ai, Bk, T(0), Bij, incRowB, incColB,
		               Ai, Bk);
	    } else {
		Backend::ugemm(mr, nr, kk, alpha, Ai, Bk, T(0), Bij, incRowB,
		               incColB, Ai, Bk);
	    }
	}
    }
}

template <typename Frame>
template <typename IndexType, typename TA, typename TB>
void
TrmmFrame<Frame>::runSerial(bool lower, IndexType m, IndexType n,
                            const T &alpha, bool conjA, bool unitDiag,
                            const TA *A, IndexType incRowA, IndexType incColA,
                            TB *B, IndexType incRowB, IndexType incColB)
{
    const std::ptrdiff_t KB = diagonalBlock(std::ptrdiff_t(m));
    static MemoryPool<T> memoryPool;

    const IndexType mcMax = std::max(IndexType(Blocks::MCmax), IndexType(KB));
    T *A_ = memoryPool.allocate(std::size_t(mcMax) * KB + MR);
    T *B_ = memoryPool.allocate(std::size_t(KB) * Blocks::NCmax + NR);

    IndexType nc, mc;
    for (IndexType j = 0; j < n; j += nc) {
	nc = Blocks::blockSize(n - j, Blocks::NC, Blocks::NCmax);

	//  the diagonal blocks: lower from the bottom, upper from the top
	//  (the last block of the lower case first, its size the rest)
	const IndexType nb = (m + KB - 1) / KB;
	for (IndexType b = 0; b < nb; ++b) {
	    const IndexType l = lower ? (nb - 1 - b) * KB : b * KB;
	    const IndexType kb = std::min(IndexType(KB), m - l);

	    TB *Bl = &B[l * incRowB + j * incColB];
	    PackB<Backend>::run(kb, nc, false, Bl, incRowB, incColB, B_);
	    PackA<Backend>::run(kb, kb, conjA, &A[l * (incRowA + incColA)],
	                        incRowA, incColA, A_);
	    prepareDiagonal(lower, unitDiag, kb, IndexType(0),
	                    IndexType((kb + MR - 1) / MR), A_);
	    multiplyBlock(lower, kb, nc, alpha, A_, B_, Bl, incRowB, incColB);

	    //  the rows below (above): B(rows) <- B(rows) + alpha A(rows,
	    //  l:l+kb) B_
	    const IndexType i0 = lower ? l + kb : IndexType(0);
	    const IndexType i1 = lower ? m : l;
	    for (IndexType i = i0; i < i1; i += mc) {
		mc = Blocks::blockSize(i1 - i, Blocks::MC, Blocks::MCmax);
		PackA<Backend>::run(mc, kb, conjA,
		                    &A[i * incRowA + l * incColA], incRowA,
		                    incColA, A_);
		Macro::run(mc, nc, kb, alpha, A_, B_, T(1),
		           &B[i * incRowB + j * incColB], incRowB, incColB);
	    }
	}
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

//
//  Threads that share the packed blocks (TrShared, shared): for a block of
//  nc columns of B, per diagonal block (rows l ... l+kb-1)
//
//      - the threads pack the diagonal block of A (some of its panels each)
//        into a common buffer and prepare its diagonal tiles, and their
//        panels of the kb rows of B into the common buffer B_; barrier;
//      - each thread multiplies its panels of the diagonal block (rows l
//        ... l+kb-1 of its columns) and updates its rectangle of the rows
//        below (above; rows and columns, whole panels, as the threads of
//        gemm), packing the block of A of its rows.
//
//  One barrier per diagonal block: the buffers alternate (two of each), a
//  buffer is packed again two blocks later, after the barrier of the block
//  between, which all threads reach only after their reads of it; the rows
//  of B packed for a block are not written by the blocks before it.
//
template <typename Frame>
template <typename IndexType, typename TA, typename TB>
void
TrmmFrame<Frame>::runShared(std::ptrdiff_t nt, bool lower, IndexType m,
                            IndexType n, const T &alpha, bool conjA,
                            bool unitDiag, const TA *A, IndexType incRowA,
                            IndexType incColA, TB *B, IndexType incRowB,
                            IndexType incColB)
{
    const std::ptrdiff_t KB = diagonalBlock(std::ptrdiff_t(m));
    static MemoryPool<T> memoryPool;

    const std::size_t sizeA = std::size_t(Blocks::MCmax) * KB + MR;
    const std::size_t sizeD = std::size_t(KB) * KB + MR;
    const std::size_t sizeB = std::size_t(KB) * Blocks::NCmax + NR;
    T *Ad = memoryPool.allocate(2 * sizeD);
    T *B_ = memoryPool.allocate(2 * sizeB);
    T *A_ = memoryPool.allocate(sizeA * std::size_t(nt));
    TreeBarrier barrier(nt);

    const IndexType nb = (m + KB - 1) / KB;

    auto body = [&](std::ptrdiff_t tid, std::ptrdiff_t threads) {
	//  fewer threads than planned (pool in use): thread 0 alone
	if (threads != nt) {
	    if (tid == 0) {
		runSerial(lower, m, n, alpha, conjA, unitDiag, A, incRowA,
		          incColA, B, incRowB, incColB);
	    }
	    return;
	}
	T *Au = A_ + sizeA * std::size_t(tid);

	std::size_t s = 0; // the number of the block (buffers s % 2)
	IndexType nc;
	for (IndexType j = 0; j < n; j += nc) {
	    nc = Blocks::blockSize(n - j, Blocks::NC, Blocks::NCmax);

	    //  this thread's panels of the nc columns
	    const IndexType np = (nc + NR - 1) / NR;
	    const IndexType c0 = std::min(
	        nc, IndexType(IndexType(tid) * np / IndexType(nt) * NR));
	    const IndexType c1 = std::min(
	        nc, IndexType(IndexType(tid + 1) * np / IndexType(nt) * NR));

	    //  lower from the bottom, upper from the top
	    for (IndexType b = 0; b < nb; ++b, ++s) {
		const IndexType l = lower ? (nb - 1 - b) * KB : b * KB;
		const IndexType kb = std::min(IndexType(KB), m - l);
		TB *Bl = &B[l * incRowB + j * incColB];
		T *Ds = Ad + sizeD * (s % 2);
		T *Bs = B_ + sizeB * (s % 2);

		//  this thread's panels of the diagonal block and of B
		const IndexType mp = (kb + MR - 1) / MR;
		const IndexType p0 = IndexType(tid) * mp / IndexType(nt);
		const IndexType p1 = IndexType(tid + 1) * mp / IndexType(nt);
		if (p1 > p0) {
		    const IndexType r0 = p0 * MR;
		    const IndexType r1 = std::min(kb, IndexType(p1 * MR));
		    PackA<Backend>::run(r1 - r0, kb, conjA,
		                        &A[(l + r0) * incRowA + l * incColA],
		                        incRowA, incColA, &Ds[r0 * kb]);
		    prepareDiagonal(lower, unitDiag, kb, p0, p1, Ds);
		}
		if (c1 > c0) {
		    PackB<Backend>::run(kb, c1 - c0, false, &Bl[c0 * incColB],
		                        incRowB, incColB, &Bs[c0 * kb]);
		}
		barrier.wait(tid);

		if (c1 > c0) {
		    multiplyBlock(lower, kb, c1 - c0, alpha, Ds, &Bs[c0 * kb],
		                  &Bl[c0 * incColB], incRowB, incColB);
		}
		const IndexType i0 = lower ? l + kb : IndexType(0);
		const IndexType i1 = lower ? m : l;
		TrShared::update<Frame>(tid, nt, i0, i1, nc, kb, alpha, conjA,
		                        &A[l * incColA], incRowA, incColA, Bs,
		                        &B[j * incColB], incRowB, incColB, Au);
	    }
	}
    };
    ThreadPool::instance().run(nt, body);

    memoryPool.release(A_);
    memoryPool.release(B_);
    memoryPool.release(Ad);
}

template <typename Frame>
template <typename IndexType, typename Alpha, typename TA, typename TB>
void
TrmmFrame<Frame>::run(bool lower, IndexType m, IndexType n, const Alpha &alpha,
                      bool conjA, bool unitDiag, const TA *A, IndexType incRowA,
                      IndexType incColA, TB *B, IndexType incRowB,
                      IndexType incColB)
{
    const std::ptrdiff_t KB = diagonalBlock(std::ptrdiff_t(m));
    if (m <= 0 || n <= 0) {
	return;
    }
    //  alpha = 0: B = 0, A is not read
    if (alpha == Alpha(0)) {
	for (IndexType j = 0; j < n; ++j) {
	    for (IndexType i = 0; i < m; ++i) {
		B[i * incRowB + j * incColB] = TB(0);
	    }
	}
	return;
    }
    const T alpha_ = T(alpha);

    //  threads: the work m^2 n / 2; the variant (TrShared)
    std::ptrdiff_t nt = GemmThreads::forProblem<T>(m, n, (m + 1) / 2);
    const std::ptrdiff_t np = (std::ptrdiff_t(n) + NR - 1) / NR;
    if (nt > 1 && TrShared::useShared("ULMBLAS_TRMM_SHARED", tuned().trmm,
                                      TunedIndex<T>::value, nt,
                                      std::ptrdiff_t(m), KB, np)) {
	runShared(nt, lower, m, n, alpha_, conjA, unitDiag, A, incRowA, incColA,
	          B, incRowB, incColB);
	return;
    }
    //  columns: at least minPanels panels (NR columns) per thread
    const std::ptrdiff_t minPanels = TrShared::minPanels(
        "ULMBLAS_TRMM_MINP", tuned().trmm, TunedIndex<T>::value);
    nt = std::min(nt, std::max(std::ptrdiff_t(1), np / minPanels));
    if (nt <= 1) {
	runSerial(lower, m, n, alpha_, conjA, unitDiag, A, incRowA, incColA, B,
	          incRowB, incColB);
	return;
    }
    auto body = [&](std::ptrdiff_t tid, std::ptrdiff_t threads) {
	//  columns of whole panels (NR); fewer threads than planned: all
	//  of them on thread 0
	const std::ptrdiff_t p0 = threads == nt ? tid * np / nt : 0;
	const std::ptrdiff_t p1 =
	    threads == nt ? (tid + 1) * np / nt : (tid == 0 ? np : 0);
	const IndexType j0 = std::min(n, IndexType(p0 * NR));
	const IndexType j1 = std::min(n, IndexType(p1 * NR));
	if (j1 > j0) {
	    runSerial(lower, m, j1 - j0, alpha_, conjA, unitDiag, A, incRowA,
	              incColA, &B[j0 * incColB], incRowB, incColB);
	}
    };
    ThreadPool::instance().run(nt, body);
}

//
//  trmm in the selection tables (gemm/plan.h): the frame of trmm with the
//  frame of the entry (for a kernel for small matrices the frame it falls
//  back to).  The problem is looked up as an m x n x m product.
//
template <typename IndexType, typename Alpha, typename TA, typename TB>
struct TrmmPlan
{
    typedef void (*Run)(bool lower, IndexType m, IndexType n,
                        const Alpha &alpha, bool conjA, bool unitDiag,
                        const TA *A, IndexType incRowA, IndexType incColA,
                        TB *B, IndexType incRowB, IndexType incColB);

    Run run;
    const char *name;
};

template <typename IndexType, typename Alpha, typename TA, typename TB>
struct TrmmOp
{
    typedef TrmmPlan<IndexType, Alpha, TA, TB> Plan;

    static constexpr std::size_t sizeC = sizeof(TB);
    static constexpr std::size_t sizeA = sizeof(TA);

    template <typename F>
    static constexpr Plan
    plan(const char *name)
    {
	typedef TrmmFrame<typename GemmFrameOf<F>::type> Frame;
	return Plan{&Frame::template run<IndexType, Alpha, TA, TB>, name};
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_FRAMETRMM_H
