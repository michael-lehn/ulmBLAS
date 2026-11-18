#ifndef ULMBLAS_LEVEL3_GEMM_FRAMETR_H
#define ULMBLAS_LEVEL3_GEMM_FRAMETR_H 1

#include <cstddef>
#include <cstdint>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level3/gemm/frame.h>
#include <ulmblas/level3/gemm/macrokerneltr.h>
#include <ulmblas/level3/gemm/pack.h>
#include <ulmblas/level3/gemm/plan.h>

//
//  The frame algorithm of gemmt: C <- beta C + alpha A B for the lower (or
//  upper) triangle of the n x n matrix C, A n x k, B k x n (k > 0, alpha !=
//  0; the special cases are handled by the wrapper, see ulmblas/level3/
//  gemmtr.tcc).  syrk and herk are gemmt with B = A^T, A^H.
//
//  The loops of gemm (ulmblas/level3/gemm/frame.h) with the backend, block
//  sizes and macro kernel of a gemm frame Frame, restricted to the
//  triangle: for a block of columns j, ..., j+nc-1 only the blocks of rows
//  that meet the triangle (lower: from row j, upper: up to row j+nc-1) are
//  packed and computed, by the macro kernel of gemmt (MacroKernelTr,
//  ulmblas/level3/gemm/macrokerneltr.h).  As BLIS's gemmt, which prunes
//  the rows outside the triangle before packing (bli_l3_prune.c).
//
//  On more threads the loops of gemm (GemmFrame::runLoops, the same
//  synchronization) with GemmTri: the groups of columns and, in each block
//  of columns, the rows of the threads have about the same number of
//  elements of the triangle (as BLIS's weighted ranges,
//  bli_thread_range_weighted_sub; here by panels, in integers).
//
namespace ulmBLAS {

//
//  Items 0, ..., count-1 with weights w(q) > 0 in parts of about the same
//  weight: part u is [b(u), b(u+1)), b(u) the first q with w(0) + ... +
//  w(q-1) >= u / parts of the total.
//
template <typename IndexType, typename Weight>
void
weightedRange(IndexType count, std::ptrdiff_t u, std::ptrdiff_t parts,
              const Weight &w, IndexType &b0, IndexType &b1)
{
    std::int64_t total = 0;
    for (IndexType q = 0; q < count; ++q) {
	total += w(q);
    }
    const std::int64_t t0 = total * u, t1 = total * (u + 1);
    std::int64_t sum = 0;
    b0 = count;
    b1 = count;
    for (IndexType q = 0; q < count; ++q) {
	if (b0 == count && sum * parts >= t0) {
	    b0 = q;
	}
	if (sum * parts >= t1) {
	    b1 = q;
	    break;
	}
	sum += w(q);
    }
    if (u + 1 == parts) {
	b1 = count;
    }
}

//
//  The part of C of gemmt for GemmFrame::runLoops: the lower (upper)
//  triangle of the n x n matrix.
//
template <typename Macro, typename IndexType> struct GemmTri
{
    bool lower;
    IndexType n, MR, NR;
    std::ptrdiff_t jcWays, P;

    //  the default packing (GemmPack)
    template <typename Backend, typename TA>
    static void
    packA(IndexType i, IndexType l, IndexType mc, IndexType kc, bool conj,
          const TA *A, IndexType incRowA, IndexType incColA,
          typename Backend::T *buffer)
    {
	GemmPack::packA<Backend>(i, l, mc, kc, conj, A, incRowA, incColA,
	                         buffer);
    }

    template <typename Backend, typename TB>
    static void
    packB(IndexType l, IndexType j, IndexType kc, IndexType nc, bool conj,
          const TB *B, IndexType incRowB, IndexType incColB,
          typename Backend::T *buffer)
    {
	GemmPack::packB<Backend>(l, j, kc, nc, conj, B, incRowB, incColB,
	                         buffer);
    }

    //  the columns of group jc: panels of NR columns weighted by their
    //  rows in the triangle
    void
    columns(std::ptrdiff_t jc, IndexType &j0, IndexType &j1) const
    {
	const IndexType np = (n + NR - 1) / NR;
	const IndexType n_ = n, NR_ = NR;
	const bool lower_ = lower;
	auto w = [=](IndexType p) -> std::int64_t {
	    const IndexType rows = lower_ ? n_ - p * NR_ : (p + 1) * NR_;
	    return rows < 1 ? 1 : (rows > n_ ? n_ : rows);
	};
	IndexType p0, p1;
	weightedRange(np, jc, jcWays, w, p0, p1);
	j0 = std::min(n, p0 * NR);
	j1 = std::min(n, p1 * NR);
    }

    //  the rows of thread u for the columns j, ..., j + nc - 1: panels of
    //  MR rows (of the rows that meet the triangle) weighted by their
    //  columns in the triangle
    void
    rows(std::ptrdiff_t u, IndexType j, IndexType nc, IndexType &i0,
         IndexType &i1) const
    {
	const IndexType r0 = lower ? j : IndexType(0);
	const IndexType r1 = lower ? n : j + nc;
	const IndexType mp = (r1 - r0 + MR - 1) / MR;
	const IndexType MR_ = MR;
	const bool lower_ = lower;
	auto w = [=](IndexType q) -> std::int64_t {
	    const IndexType r = r0 + q * MR_ + MR_ / 2;
	    const IndexType cols = lower_ ? r - j + 1 : j + nc - r;
	    return cols < 1 ? 1 : (cols > nc ? nc : cols);
	};
	IndexType q0, q1;
	weightedRange(mp, u, P, w, q0, q1);
	i0 = std::min(r1, r0 + q0 * MR);
	i1 = std::min(r1, r0 + q1 * MR);
    }

    IndexType
    maxRows() const
    {
	return n;
    }

    template <typename T, typename Beta, typename TC>
    void
    macro(IndexType i, IndexType j, IndexType mc, IndexType nc, IndexType kc,
          const T &alpha, const T *A_, const T *B_, const Beta &beta, TC *C,
          IndexType incRowC, IndexType incColC) const
    {
	Macro::run(lower, i - j, mc, nc, kc, alpha, A_, B_, beta,
	           &C[i * incRowC + j * incColC], incRowC, incColC);
    }
};

template <typename Frame> struct GemmtFrame
{
    typedef typename Frame::BackendType Backend;
    typedef typename Frame::BlocksType Blocks;
    typedef MacroKernelTr<Backend, typename Frame::MacroType> Macro;
    typedef typename Backend::T T;

    template <typename IndexType, typename Alpha, typename TA, typename TB,
              typename Beta, typename TC>
    static void
    run(bool lower, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
        const TA *A, IndexType incRowA, IndexType incColA, bool conjB,
        const TB *B, IndexType incRowB, IndexType incColB, const Beta &beta,
        TC *C, IndexType incRowC, IndexType incColC);

    //  the loops on one thread
    template <typename IndexType, typename Alpha, typename TA, typename TB,
              typename Beta, typename TC>
    static void
    runSerial(bool lower, IndexType n, IndexType k, const Alpha &alpha,
              bool conjA, const TA *A, IndexType incRowA, IndexType incColA,
              bool conjB, const TB *B, IndexType incRowB, IndexType incColB,
              const Beta &beta, TC *C, IndexType incRowC, IndexType incColC);
};

template <typename Frame>
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
GemmtFrame<Frame>::run(bool lower, IndexType n, IndexType k, const Alpha &alpha,
                       bool conjA, const TA *A, IndexType incRowA,
                       IndexType incColA, bool conjB, const TB *B,
                       IndexType incRowB, IndexType incColB, const Beta &beta,
                       TC *C, IndexType incRowC, IndexType incColC)
{
    //  kernels that prefer row-major C: for column-major C compute C^T =
    //  B^T A^T, the other triangle
    if (Backend::rowPreference && incRowC == IndexType(1) &&
        incColC != IndexType(1)) {
	run(!lower, n, k, alpha, conjB, B, incColB, incRowB, conjA, A, incColA,
	    incRowA, beta, C, incColC, incRowC);
	return;
    }
    auto serial = [&] {
	runSerial(lower, n, k, alpha, conjA, A, incRowA, incColA, conjB, B,
	          incRowB, incColB, beta, C, incRowC, incColC);
    };
    //  the threads for the work of the triangle, n (n + 1) / 2 k
    const std::ptrdiff_t nt = GemmThreads::forProblem<T>(n, (n + 1) / 2, k);
    std::ptrdiff_t jcWays = 1, P = 1;
    if (nt > 1) {
	Frame::partitionThreads(nt, std::ptrdiff_t(n), std::ptrdiff_t(n),
	                        jcWays, P);
    }
    if (jcWays * P <= 1) {
	serial();
	return;
    }
    const GemmTri<Macro, IndexType> shape = {
        lower, n, IndexType(Backend::MR), IndexType(Backend::NR), jcWays, P};
    Frame::runLoops(jcWays, P, shape, serial, k, alpha, conjA, A, incRowA,
                    incColA, conjB, B, incRowB, incColB, beta, C, incRowC,
                    incColC);
}

template <typename Frame>
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
GemmtFrame<Frame>::runSerial(bool lower, IndexType n, IndexType k,
                             const Alpha &alpha, bool conjA, const TA *A,
                             IndexType incRowA, IndexType incColA, bool conjB,
                             const TB *B, IndexType incRowB, IndexType incColB,
                             const Beta &beta, TC *C, IndexType incRowC,
                             IndexType incColC)
{
    constexpr IndexType MR = Backend::MR;
    constexpr IndexType NR = Backend::NR;

    static MemoryPool<T> memoryPool;

    T *A_ = memoryPool.allocate(Blocks::MCmax * Blocks::KCmax + MR);
    T *B_ = memoryPool.allocate(Blocks::KCmax * Blocks::NCmax + NR);

    IndexType nc, kc, mc;
    for (IndexType j = 0; j < n; j += nc) {
	nc = Blocks::blockSize(n - j, Blocks::NC, Blocks::NCmax);

	//  the rows that meet the triangle in these columns
	const IndexType i0 = lower ? j : IndexType(0);
	const IndexType i1 = lower ? n : j + nc;

	for (IndexType l = 0; l < k; l += kc) {
	    kc = Blocks::blockSize(k - l, Blocks::KC, Blocks::KCmax);
	    Beta beta_ = (l == 0) ? beta : Beta(1);

	    PackB<Backend>::run(kc, nc, conjB, &B[l * incRowB + j * incColB],
	                        incRowB, incColB, B_);

	    for (IndexType i = i0; i < i1; i += mc) {
		mc = Blocks::blockSize(i1 - i, Blocks::MC, Blocks::MCmax);

		PackA<Backend>::run(mc, kc, conjA,
		                    &A[i * incRowA + l * incColA], incRowA,
		                    incColA, A_);
		Macro::run(lower, i - j, mc, nc, kc, alpha, A_, B_, beta_,
		           &C[i * incRowC + j * incColC], incRowC, incColC);
	    }
	}
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

//
//  gemmt in the selection tables (gemm/plan.h): the plan is the frame of
//  gemmt with the frame of the entry (for a kernel for small matrices the
//  frame it falls back to).
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
struct GemmtPlan
{
    typedef void (*Run)(bool lower, IndexType n, IndexType k,
                        const Alpha &alpha, bool conjA, const TA *A,
                        IndexType incRowA, IndexType incColA, bool conjB,
                        const TB *B, IndexType incRowB, IndexType incColB,
                        const Beta &beta, TC *C, IndexType incRowC,
                        IndexType incColC);

    Run run;
    const char *name;
};

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
struct GemmtOp
{
    typedef GemmtPlan<IndexType, Alpha, TA, TB, Beta, TC> Plan;

    static constexpr std::size_t sizeC = sizeof(TC);
    static constexpr std::size_t sizeA = sizeof(TA);

    template <typename F>
    static Plan
    plan(const char *name)
    {
	typedef GemmtFrame<typename GemmFrameOf<F>::type> Frame;
	Plan p = {&Frame::template run<IndexType, Alpha, TA, TB, Beta, TC>,
	          name};
	return p;
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_FRAMETR_H
