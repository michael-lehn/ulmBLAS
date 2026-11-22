#ifndef ULMBLAS_LEVEL3_GEMM_FRAME_H
#define ULMBLAS_LEVEL3_GEMM_FRAME_H 1

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/level3/gemm/threads.h>
#include <ulmblas/level3/gemm/macrokernel.h>
#include <ulmblas/level3/gemm/pack.h>

//
//  The frame algorithm of gemm: C <- beta C + alpha A B for an m x n matrix
//  C (k > 0, alpha != 0; the special cases are handled by the wrapper, see
//  ulmblas/level3/gemm.tcc).
//
//  The loops over the blocks: for each block of columns of C (nc columns),
//  for each block of columns of A (rows of B, kc), pack the block of B (kc x
//  nc) once, then for each block of rows of C (mc rows) pack the block of A
//  (mc x kc) and call the macro kernel for the block of C (mc x nc):
//
//      for j = 0, nc, ...                blocks of columns of C and B
//          for l = 0, kc, ...            blocks of columns of A, rows of B
//              B_ <- pack(B(l:l+kc, j:j+nc))
//              for i = 0, mc, ...        blocks of rows of C and A
//                  A_ <- pack(A(i:i+mc, l:l+kc))
//                  C(i:i+mc, j:j+nc) <- beta_ C(...) + alpha A_ B_
//
//  with beta_ = beta for the first block of k and 1 for the others.  The
//  block sizes are those of GemmBlocks (see there): the block size, or the
//  whole rest of a dimension if it is at most the maximum block size.
//
//  Template parameters (compile-time constants):
//
//      Backend      micro kernel and packing of single panels
//                   (ulmblas/level3/ukernel/backend.h); its type T is the
//                   type of the packed blocks (A and B are converted to T
//                   while packing)
//      Blocks       block sizes, GemmBlocks<MC, KC, NC, MCmax, KCmax,
//                   NCmax>
//      Interleave   0: pack the whole block of B before the loop over i.
//                   p > 0: pack B for the first block of A (i = 0) p panels
//                   (p NR columns) at a time, each used right away by the
//                   macro kernel while it is in cache (as OpenBLAS's
//                   driver).  The packed layout is the same.
//      Macro        the macro kernel (default: MacroKernel<Backend>,
//                   ulmblas/level3/gemm/macrokernel.h)
//
namespace ulmBLAS {

//
//  Block sizes of the frame algorithm: MC rows of A and C, KC columns of A
//  (rows of B), NC columns of B and C (MC a multiple of the backend's MR,
//  NC of NR, for full panels), and the maximum block sizes MCmax, KCmax,
//  NCmax (default: no larger than the block sizes).
//
//  A dimension is split into blocks of the block size, except that the
//  rest is taken as a whole if it is at most the maximum: with NC = 256,
//  NCmax = 264 (NC + NR for NR = 8), n = 520 is split into 264 + 256, not
//  256 + 256 + 8 (a block of a single panel).  As in BLIS, where block sizes
//  have a default and a maximum value.  The buffers for the packed blocks
//  have the maximum sizes.
//
template <std::ptrdiff_t MC_, std::ptrdiff_t KC_, std::ptrdiff_t NC_,
          std::ptrdiff_t MCmax_ = MC_, std::ptrdiff_t KCmax_ = KC_,
          std::ptrdiff_t NCmax_ = NC_>
struct GemmBlocks
{
    static_assert(MCmax_ >= MC_ && KCmax_ >= KC_ && NCmax_ >= NC_,
                  "GemmBlocks: maximum block sizes below the block sizes");

    static constexpr std::ptrdiff_t MC = MC_, KC = KC_, NC = NC_;
    static constexpr std::ptrdiff_t MCmax = MCmax_, KCmax = KCmax_,
                                    NCmax = NCmax_;

    //  The size of the next block with rest elements of a dimension left
    template <typename IndexType>
    static IndexType
    blockSize(IndexType rest, std::ptrdiff_t block, std::ptrdiff_t max)
    {
	return rest <= IndexType(max) ? rest : IndexType(block);
    }
};

template <typename Backend, typename Blocks, std::ptrdiff_t Interleave = 0,
          typename Macro = MacroKernel<Backend>>
struct GemmFrame
{
    typedef typename Backend::T T;

    //  the parameters (for the frame of gemmt, ulmblas/level3/gemm/
    //  frametr.h)
    typedef Backend BackendType;
    typedef Blocks BlocksType;
    typedef Macro MacroType;

    //  the buffers hold blocks of whole panels: a block of the maximal size
    //  fills them exactly (see runSerial, runParallel)
    static_assert(Blocks::MCmax % Backend::MR == 0,
                  "GemmFrame: MCmax must be a multiple of MR");
    static_assert(Blocks::NCmax % Backend::NR == 0,
                  "GemmFrame: NCmax must be a multiple of NR");

    template <typename IndexType, typename Alpha, typename TA, typename TB,
              typename Beta, typename TC>
    static void
    run(IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
        const TA *A, IndexType incRowA, IndexType incColA, bool conjB,
        const TB *B, IndexType incRowB, IndexType incColB, const Beta &beta,
        TC *C, IndexType incRowC, IndexType incColC);

    //  the loops on one thread
    template <typename IndexType, typename Alpha, typename TA, typename TB,
              typename Beta, typename TC>
    static void
    runSerial(IndexType m, IndexType n, IndexType k, const Alpha &alpha,
              bool conjA, const TA *A, IndexType incRowA, IndexType incColA,
              bool conjB, const TB *B, IndexType incRowB, IndexType incColB,
              const Beta &beta, TC *C, IndexType incRowC, IndexType incColC);

    //  the loops on nt threads (see GemmThreads)
    template <typename IndexType, typename Alpha, typename TA, typename TB,
              typename Beta, typename TC>
    static void
    runParallel(std::ptrdiff_t nt, IndexType m, IndexType n, IndexType k,
                const Alpha &alpha, bool conjA, const TA *A, IndexType incRowA,
                IndexType incColA, bool conjB, const TB *B, IndexType incRowB,
                IndexType incColB, const Beta &beta, TC *C, IndexType incRowC,
                IndexType incColC);

    //  nt threads for an m x n matrix C: jcWays groups of P threads (see
    //  runParallel)
    static void
    partitionThreads(std::ptrdiff_t nt, std::ptrdiff_t m, std::ptrdiff_t n,
                     std::ptrdiff_t &jcWays, std::ptrdiff_t &P);

    //  the loops of runParallel for the part of C that shape describes
    //  (GemmRect below: all of C; gemmt: a triangle, ulmblas/level3/gemm/
    //  frametr.h); serial() computes the problem on the calling thread
    template <typename Shape, typename Serial, typename IndexType,
              typename Alpha, typename TA, typename TB, typename Beta,
              typename TC>
    static void
    runLoops(std::ptrdiff_t jcWays, std::ptrdiff_t P, const Shape &shape,
             const Serial &serial, IndexType k, const Alpha &alpha, bool conjA,
             const TA *A, IndexType incRowA, IndexType incColA, bool conjB,
             const TB *B, IndexType incRowB, IndexType incColB,
             const Beta &beta, TC *C, IndexType incRowC, IndexType incColC);
};

template <typename Backend, typename Blocks, std::ptrdiff_t Interleave,
          typename Macro>
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
GemmFrame<Backend, Blocks, Interleave, Macro>::run(
    IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
    const TA *A, IndexType incRowA, IndexType incColA, bool conjB, const TB *B,
    IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
    IndexType incRowC, IndexType incColC)
{
    //  kernels that prefer row-major C: for column-major C compute
    //  C^T = B^T A^T
    if (Backend::rowPreference && incRowC == IndexType(1) &&
        incColC != IndexType(1)) {
	run(n, m, k, alpha, conjB, B, incColB, incRowB, conjA, A, incColA,
	    incRowA, beta, C, incColC, incRowC);
	return;
    }
    const std::ptrdiff_t nt = GemmThreads::forProblem<T>(m, n, k);
    if (nt > 1) {
	runParallel(nt, m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B,
	            incRowB, incColB, beta, C, incRowC, incColC);
    } else {
	runSerial(m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B, incRowB,
	          incColB, beta, C, incRowC, incColC);
    }
}

template <typename Backend, typename Blocks, std::ptrdiff_t Interleave,
          typename Macro>
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
GemmFrame<Backend, Blocks, Interleave, Macro>::runSerial(
    IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
    const TA *A, IndexType incRowA, IndexType incColA, bool conjB, const TB *B,
    IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
    IndexType incRowC, IndexType incColC)
{
    constexpr IndexType MR = Backend::MR;
    constexpr IndexType NR = Backend::NR;

    static MemoryPool<T> memoryPool;

    T *A_ = memoryPool.allocate(Blocks::MCmax * Blocks::KCmax + MR);
    T *B_ = memoryPool.allocate(Blocks::KCmax * Blocks::NCmax + NR);

    IndexType nc, kc, mc;
    for (IndexType j = 0; j < n; j += nc) {
	nc = Blocks::blockSize(n - j, Blocks::NC, Blocks::NCmax);

	for (IndexType l = 0; l < k; l += kc) {
	    kc = Blocks::blockSize(k - l, Blocks::KC, Blocks::KCmax);
	    Beta beta_ = (l == 0) ? beta : Beta(1);

	    if (Interleave == 0) {
		PackB<Backend>::run(kc, nc, conjB,
		                    &B[l * incRowB + j * incColB], incRowB,
		                    incColB, B_);
	    }

	    for (IndexType i = 0; i < m; i += mc) {
		mc = Blocks::blockSize(m - i, Blocks::MC, Blocks::MCmax);

		PackA<Backend>::run(mc, kc, conjA,
		                    &A[i * incRowA + l * incColA], incRowA,
		                    incColA, A_);

		if (Interleave > 0 && i == 0) {
		    //  pack B a few panels at a time, each used right away
		    constexpr IndexType NI =
		        NR * (Interleave > 0 ? Interleave : 1);
		    for (IndexType jr = 0; jr < nc; jr += NI) {
			IndexType nr = (nc - jr < NI) ? nc - jr : NI;

			PackB<Backend>::run(
			    kc, nr, conjB, &B[l * incRowB + (j + jr) * incColB],
			    incRowB, incColB, &B_[jr * kc]);
			Macro::run(mc, nr, kc, alpha, A_, &B_[jr * kc], beta_,
			           &C[(j + jr) * incColC], incRowC, incColC);
		    }
		} else {
		    Macro::run(mc, nc, kc, alpha, A_, B_, beta_,
		               &C[i * incRowC + j * incColC], incRowC, incColC);
		}
	    }
	}
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

//
//  The loops on nt threads, with the synchronization of OpenBLAS's
//  level3_thread.c (ready flags instead of barriers).  The partition
//  (GemmThreads::partition as BLIS if the threads are pinned,
//  GemmThreads::partitionOpenBLAS otherwise) gives jcWays groups for ranges of
//  columns of C (whole panels of NR columns) of P = icWays jrWays threads each.
//  The threads of a group go through the same blocks of columns (nc) and of k
//  (kc); thread u of a group
//
//      - packs the blocks of A of its rows (whole panels of MR rows, rows
//        u mp/P to (u+1) mp/P of the mp panels) into its own buffer,
//      - packs its share of the block of B (panels u bp/P to (u+1) bp/P of
//        the bp panels) into the group's buffer, sets its ready flag and
//        uses it right away,
//      - computes its rows of C with the other shares of B, each as soon
//        as its ready flag is set, and then counts the block as finished.
//
//  The group has two buffers for B, alternately: before a thread packs
//  into one again, it waits until all threads of the group have finished
//  the block from two blocks before (a counter of finished blocks per
//  thread; a sum of uses would let a thread that runs ahead count for
//  one that is late).  No barriers: a thread that is late (e.g.
//  sharing a core with another thread) delays only the threads that wait
//  for its share.  If the pool runs the region on fewer threads (nested
//  call, pool in use), runSerial on thread 0.
//
//  Correctness (each counter has one writer and increases):
//
//      - a share is read only after it is packed: the reader waits for
//        ready >= s + 1, stored after the packing (release/acquire);
//      - a buffer is packed again only after all reads of it: the packer
//        waits for done >= s - 1 of every thread of the group, stored
//        after its reads of block s - 2 (release/acquire);
//      - no deadlock: in a block a thread packs all its shares before it
//        waits for others; by induction over the blocks every wait ends;
//      - the threads write disjoint rectangles of C, A buffers are
//        private.
//
//  test/protocol/frame.py (make check-protocol) checks the protocol for all
//  interleavings of small cases.
//
//  a counter of the parallel frame (a cache line of its own): of a share
//  of B, the number of blocks packed into it; of a thread, the number of
//  blocks it has finished.  Aligned to 128 bytes (the blocks of MemoryPool
//  have the alignment of the type) and padded explicitly, so that the
//  compiler adds no padding (no warning, e.g. MSVC C4324).
struct alignas(128) GemmFlag
{
    std::atomic<std::ptrdiff_t> value;
    char pad[128 - sizeof(std::atomic<std::ptrdiff_t>)];

    GemmFlag() : value(0) {}

    void
    waitFor(std::ptrdiff_t v) const
    {
	threads::Backoff backoff;
	while (value.load(std::memory_order_acquire) < v) {
	    backoff.pause();
	}
    }
};

static_assert(sizeof(GemmFlag) == 128, "GemmFlag: one 128-byte block");

//
//  The packing of the shapes of runLoops: the block of A of rows i, ...,
//  i + mc - 1 and columns l, ..., l + kc - 1, the block of B of rows l, ...,
//  l + kc - 1 and columns j, ..., j + nc - 1 (whole matrices A and B with
//  the position of the block: symm packs a symmetric matrix from its
//  stored triangle, ulmblas/level3/gemm/framesymm.h).
//
struct GemmPack
{
    template <typename Backend, typename IndexType, typename TA>
    static void
    packA(IndexType i, IndexType l, IndexType mc, IndexType kc, bool conj,
          const TA *A, IndexType incRowA, IndexType incColA,
          typename Backend::T *buffer)
    {
	PackA<Backend>::run(mc, kc, conj, &A[i * incRowA + l * incColA],
	                    incRowA, incColA, buffer);
    }

    template <typename Backend, typename IndexType, typename TB>
    static void
    packB(IndexType l, IndexType j, IndexType kc, IndexType nc, bool conj,
          const TB *B, IndexType incRowB, IndexType incColB,
          typename Backend::T *buffer)
    {
	PackB<Backend>::run(kc, nc, conj, &B[l * incRowB + j * incColB],
	                    incRowB, incColB, buffer);
    }
};

//
//  The part of C of gemm for runLoops: all of the m x n matrix, groups of
//  columns and rows of the threads of equal size (whole panels), the macro
//  kernel of gemm.
//
template <typename Macro, typename IndexType> struct GemmRect
{
    IndexType m, n, MR, NR;
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

    //  the columns of group jc
    void
    columns(std::ptrdiff_t jc, IndexType &j0, IndexType &j1) const
    {
	const IndexType np = (n + NR - 1) / NR;
	j0 = std::min(n, IndexType(jc * np / jcWays) * NR);
	j1 = std::min(n, IndexType((jc + 1) * np / jcWays) * NR);
    }

    //  the rows of thread u of a group for the block of columns j, ...,
    //  j + nc - 1
    void
    rows(std::ptrdiff_t u, IndexType j, IndexType nc, IndexType &i0,
         IndexType &i1) const
    {
	const IndexType mp = (m + MR - 1) / MR;
	i0 = std::min(m, IndexType(u * mp / P) * MR);
	i1 = std::min(m, IndexType((u + 1) * mp / P) * MR);
    }

    //  the most rows of a thread in a block
    IndexType
    maxRows() const
    {
	const IndexType mp = (m + MR - 1) / MR;
	return (mp + IndexType(P) - 1) / IndexType(P) * MR;
    }

    template <typename T, typename Beta, typename TC>
    void
    macro(IndexType i, IndexType j, IndexType mc, IndexType nc, IndexType kc,
          const T &alpha, const T *A_, const T *B_, const Beta &beta, TC *C,
          IndexType incRowC, IndexType incColC) const
    {
	Macro::run(mc, nc, kc, alpha, A_, B_, beta,
	           &C[i * incRowC + j * incColC], incRowC, incColC);
    }
};

template <typename Backend, typename Blocks, std::ptrdiff_t Interleave,
          typename Macro>
void
GemmFrame<Backend, Blocks, Interleave, Macro>::partitionThreads(
    std::ptrdiff_t nt, std::ptrdiff_t m, std::ptrdiff_t n,
    std::ptrdiff_t &jcWays, std::ptrdiff_t &P)
{
    constexpr std::ptrdiff_t MR = Backend::MR;
    constexpr std::ptrdiff_t NR = Backend::NR;

    std::ptrdiff_t icWays, jrWays;
    //  unpinned threads: the partition of OpenBLAS (more ranges of rows,
    //  which suits threads the operating system moves)
    if (threads::Affinity::instance().mode.load(std::memory_order_relaxed) ==
        0) {
	//  16 for float and complex float, 8 for double and complex double
	const std::ptrdiff_t switchRatio =
	    sizeof(T) == 4 ||
	            (sizeof(T) == 8 && !std::is_floating_point<T>::value)
	        ? 16
	        : 8;
	GemmThreads::partitionOpenBLAS(nt, m, n, switchRatio, jcWays, icWays);
	jrWays = 1;
	const std::ptrdiff_t mp = (m + MR - 1) / MR, np = (n + NR - 1) / NR;
	icWays = std::min(icWays, mp);
	jcWays = std::min(jcWays, np);
    } else {
	GemmThreads::partition(nt, m, n, MR, NR, jcWays, icWays, jrWays);
    }
    //  the threads of a group, each packs one share of B
    P = icWays * jrWays;
}

template <typename Backend, typename Blocks, std::ptrdiff_t Interleave,
          typename Macro>
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
GemmFrame<Backend, Blocks, Interleave, Macro>::runParallel(
    std::ptrdiff_t nt, IndexType m, IndexType n, IndexType k,
    const Alpha &alpha, bool conjA, const TA *A, IndexType incRowA,
    IndexType incColA, bool conjB, const TB *B, IndexType incRowB,
    IndexType incColB, const Beta &beta, TC *C, IndexType incRowC,
    IndexType incColC)
{
    std::ptrdiff_t jcWays, P;
    partitionThreads(nt, std::ptrdiff_t(m), std::ptrdiff_t(n), jcWays, P);
    auto serial = [&] {
	runSerial(m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B, incRowB,
	          incColB, beta, C, incRowC, incColC);
    };
    if (jcWays * P <= 1) {
	serial();
	return;
    }
    const GemmRect<Macro, IndexType> shape = {
        m, n, IndexType(Backend::MR), IndexType(Backend::NR), jcWays, P};
    runLoops(jcWays, P, shape, serial, k, alpha, conjA, A, incRowA, incColA,
             conjB, B, incRowB, incColB, beta, C, incRowC, incColC);
}

template <typename Backend, typename Blocks, std::ptrdiff_t Interleave,
          typename Macro>
template <typename Shape, typename Serial, typename IndexType, typename Alpha,
          typename TA, typename TB, typename Beta, typename TC>
void
GemmFrame<Backend, Blocks, Interleave, Macro>::runLoops(
    std::ptrdiff_t jcWays, std::ptrdiff_t P, const Shape &shape,
    const Serial &serial, IndexType k, const Alpha &alpha, bool conjA,
    const TA *A, IndexType incRowA, IndexType incColA, bool conjB, const TB *B,
    IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
    IndexType incRowC, IndexType incColC)
{
    constexpr IndexType MR = Backend::MR;
    constexpr IndexType NR = Backend::NR;

    static MemoryPool<T> memoryPool;
    static MemoryPool<GemmFlag> flagsPool;

    const std::ptrdiff_t nt = jcWays * P;

    //  buffers: two for B per group, one for A per thread, for the blocks
    //  of the groups' columns and the threads' rows (at most the maximal
    //  blocks)
    std::size_t ncg = 0;
    for (std::ptrdiff_t jc = 0; jc < jcWays; ++jc) {
	IndexType j0, j1;
	shape.columns(jc, j0, j1);
	ncg = std::max(ncg, std::size_t(j1 - j0));
    }
    ncg = std::min(std::size_t(Blocks::NCmax), (ncg + std::size_t(NR) - 1) /
                                                   std::size_t(NR) *
                                                   std::size_t(NR));
    //  whole panels: the last one is padded to MR rows
    const std::size_t mcg =
        std::min(std::size_t(Blocks::MCmax),
	         (std::size_t(shape.maxRows()) + std::size_t(MR) - 1) /
	             std::size_t(MR) * std::size_t(MR));
    const std::size_t kcg =
        std::min(std::size_t(Blocks::KCmax), std::size_t(k));
    const std::size_t sizeB = kcg * ncg + NR;
    const std::size_t sizeA = mcg * kcg + MR;
    T *B_ = memoryPool.allocate(2 * sizeB * std::size_t(jcWays));
    T *A_ = memoryPool.allocate(sizeA * std::size_t(nt));

    //  the counters, per group P for the shares (ready) and P for the
    //  threads (done); in memory of a pool (nothing here throws; trivially
    //  destructible members)
    GemmFlag *flags = flagsPool.allocate(std::size_t(jcWays * 2 * P));
    for (std::ptrdiff_t t = 0; t < jcWays * 2 * P; ++t) {
	new (&flags[t]) GemmFlag();
    }

    auto body = [&](std::ptrdiff_t tid, std::ptrdiff_t threads) {
	//  fewer threads than the partition (pool in use, workers missing):
	//  thread 0 alone
	if (threads != nt) {
	    if (tid == 0) {
		serial();
	    }
	    return;
	}
	const std::ptrdiff_t jc = tid / P;
	const std::ptrdiff_t u = tid % P;
	GemmFlag *ready = flags + jc * 2 * P;
	GemmFlag *done = ready + P;
	T *Bg = B_ + 2 * sizeB * std::size_t(jc);
	T *Au = A_ + sizeA * std::size_t(tid);

	//  the columns of the group
	IndexType j0, j1;
	shape.columns(jc, j0, j1);

	std::ptrdiff_t s = 0; // the number of the block (j, l) of the group
	IndexType nc, kc, mc;
	for (IndexType j = j0; j < j1; j += nc) {
	    nc = Blocks::blockSize(j1 - j, Blocks::NC, Blocks::NCmax);
	    const IndexType bp = (nc + NR - 1) / NR;

	    //  the rows of the thread in these columns
	    IndexType i0, i1;
	    shape.rows(u, j, nc, i0, i1);

	    for (IndexType l = 0; l < k; l += kc, ++s) {
		kc = Blocks::blockSize(k - l, Blocks::KC, Blocks::KCmax);
		Beta beta_ = (l == 0) ? beta : Beta(1);
		T *Bs = Bg + sizeB * std::size_t(s & 1);

		//  the buffer is free when every thread of the group has
		//  finished the block before the last (s - 2), which used it
		if (s >= 2) {
		    for (std::ptrdiff_t v = 0; v < P; ++v) {
			done[v].waitFor(s - 1);
		    }
		}

		//  share c of B (this thread's), once its buffer is free;
		//  packed in every block, also by a thread without rows in
		//  it (the others wait for it)
		auto packShare = [&](std::ptrdiff_t c) {
		    const IndexType p0 = IndexType(c) * bp / IndexType(P);
		    const IndexType p1 = IndexType(c + 1) * bp / IndexType(P);
		    if (p1 > p0) {
			const IndexType c0 = p0 * NR;
			const IndexType c1 = std::min(nc, p1 * NR);
			shape.template packB<Backend>(
			    l, j + c0, kc, c1 - c0, conjB, B, incRowB, incColB,
			    Bs + std::size_t(p0) * kc * NR);
		    }
		    ready[c].value.store(s + 1, std::memory_order_release);
		};
		if (i0 == i1) {
		    packShare(u);
		}

		//  this thread's rows with all shares: in the first block of
		//  rows its own share first, packed and used right away (as
		//  OpenBLAS), then the others as they are ready
		for (IndexType i = i0; i < i1; i += mc) {
		    mc = Blocks::blockSize(i1 - i, Blocks::MC, Blocks::MCmax);
		    shape.template packA<Backend>(i, l, mc, kc, conjA, A,
		                                  incRowA, incColA, Au);
		    for (std::ptrdiff_t t = 0; t < P; ++t) {
			const std::ptrdiff_t c = (u + t) % P;
			if (i == i0) {
			    if (t == 0) {
				packShare(c);
			    } else {
				ready[c].waitFor(s + 1);
			    }
			}
			const IndexType p0 = IndexType(c) * bp / IndexType(P);
			const IndexType p1 =
			    IndexType(c + 1) * bp / IndexType(P);
			if (p1 <= p0) {
			    continue;
			}
			const IndexType c0 = p0 * NR;
			const IndexType c1 = std::min(nc, p1 * NR);
			shape.macro(i, j + c0, mc, c1 - c0, kc, alpha, Au,
			            Bs + std::size_t(p0) * kc * NR, beta_, C,
			            incRowC, incColC);
		    }
		}
		done[u].value.store(s + 1, std::memory_order_release);
	    }
	}
    };
    ThreadPool::instance().run(nt, body);

    flagsPool.release(flags);
    memoryPool.release(A_);
    memoryPool.release(B_);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_FRAME_H
