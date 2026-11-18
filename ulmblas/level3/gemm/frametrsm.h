#ifndef ULMBLAS_LEVEL3_GEMM_FRAMETRSM_H
#define ULMBLAS_LEVEL3_GEMM_FRAMETRSM_H 1

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/config/inline.h>
#include <ulmblas/config/platform.h>
#include <ulmblas/config/tuned.h>
#include <ulmblas/level3/gemm/frame.h>
#include <ulmblas/level3/gemm/pack.h>
#include <ulmblas/level3/gemm/plan.h>
#include <ulmblas/level3/gemm/threads.h>
#include <ulmblas/level3/gemm/trshared.h>

//
//  The frame algorithm of trsm: B <- alpha A^{-1} B for a lower (upper)
//  triangular m x m matrix A, B m x n, with the backend, block sizes and
//  macro kernel of a gemm frame Frame (ulmblas/level3/gemm/frame.h).  The
//  other cases (A on the right, op(A) = A^T, A^H) are reduced to these two
//  by the BLAS interface (strides of A and B swapped, see
//  interfaces/blas/C/trsm.cc).  Alpha scales B first (as OpenBLAS).
//
//  As OpenBLAS's driver (driver/level3/trsm_L.c) with its generic trsm
//  kernel (kernel/generic/trsm_kernel_LT.c): for a block of columns of B
//  (nc), the diagonal blocks of A (kb x kb) one after the other (lower:
//  from the top, upper: from the bottom).  For a diagonal block
//
//      - the block of B (its kb rows) is packed (B_), the diagonal block of
//        A is packed (A_) and its diagonal tiles (MR x MR) prepared: the
//        diagonal inverted (1 for a unit diagonal), zeros on the other
//        side of the diagonal;
//      - tile by tile (MR x NR, in the order of the substitution): the
//        micro kernel subtracts the product of the tile's rows of A with
//        the rows of X already solved in this block (from B_), then a
//        substitution with the MR x MR diagonal tile solves the tile; the
//        solution goes to B and to B_;
//      - the rows of B below (above) the block are updated with B_, the
//        solution of the block: C <- C - A(rows, block) X(block) with the
//        macro kernel of the frame (gemm).
//
//  The diagonal blocks have at most KB rows (as OpenBLAS's GEMM_Q, 256 to
//  384): with the KC of gemm (up to 2800 on an Apple M5) most of the work
//  would be in the substitution and the short products of the diagonal
//  blocks.
//
//  On nt threads (GemmThreads::forProblem for the work m^2 n / 2) one of
//  two variants (TrShared, ulmblas/level3/gemm/trshared.h): the columns of
//  B split (as OpenBLAS's gemm_thread_n; every thread solves its columns
//  independently, with its own buffers, no synchronization), or threads
//  that share the packed blocks (runShared, as BLIS).
//
namespace ulmBLAS {

template <typename Frame> struct TrsmFrame
{
    typedef typename Frame::BackendType Backend;
    typedef typename Frame::BlocksType Blocks;
    typedef typename Frame::MacroType Macro;
    typedef typename Backend::T T;

    static constexpr std::ptrdiff_t MR = Backend::MR;
    static constexpr std::ptrdiff_t NR = Backend::NR;

    //  the largest diagonal block for an m x m A: tuned().trsm.kb(m)
    //  (ulmblas/config/tuned.h, by default 256) rounded up to whole panels,
    //  at most KCmax (whole panels) and 2048, at least one panel (test
    //  configurations with KCmax < MR)
    static std::ptrdiff_t
    diagonalBlock(std::ptrdiff_t m)
    {
	constexpr std::ptrdiff_t KC1 = Blocks::KCmax / MR * MR;
	constexpr std::ptrdiff_t KBmax = KC1 < 2048 ? KC1 : 2048 / MR * MR;
	const std::ptrdiff_t kb =
	    std::ptrdiff_t(tuned().trsm.kb[TunedIndex<T>::value](m));
	const std::ptrdiff_t KB = (kb + MR - 1) / MR * MR;
	return KB < MR ? MR : (KB < KBmax ? KB : (KBmax >= MR ? KBmax : MR));
    }

    template <typename IndexType, typename Alpha, typename TA, typename TB>
    static void
    run(bool lower, IndexType m, IndexType n, const Alpha &alpha, bool conjA,
        bool unitDiag, const TA *A, IndexType incRowA, IndexType incColA, TB *B,
        IndexType incRowB, IndexType incColB);

    //  the loops on one thread (B already scaled by alpha)
    template <typename IndexType, typename TA, typename TB>
    static void
    runSerial(bool lower, IndexType m, IndexType n, bool conjA, bool unitDiag,
              const TA *A, IndexType incRowA, IndexType incColA, TB *B,
              IndexType incRowB, IndexType incColB);

  private:
    //  threads that share the packed blocks of A (see run)
    template <typename IndexType, typename TA, typename TB>
    static void
    runShared(std::ptrdiff_t nt, bool lower, IndexType m, IndexType n,
              bool conjA, bool unitDiag, const TA *A, IndexType incRowA,
              IndexType incColA, TB *B, IndexType incRowB, IndexType incColB);

    //  the panels p0, ..., p1-1 of the packed diagonal block A_ prepared
    template <typename IndexType>
    static void
    prepareDiagonal(bool lower, bool unitDiag, IndexType kb, IndexType p0,
                    IndexType p1, T *A_, T *Inv);

    template <typename IndexType, typename TB>
    static void
    solveBlock(bool lower, IndexType kb, IndexType nc, const T *A_,
               const T *Inv, T *B_, TB *B, IndexType incRowB,
               IndexType incColB);

    //  1 / a, not inlined: clang 19 with AVX-512 vectorized the loop over
    //  the diagonal of a tile with a masked gather (zeros in the lanes
    //  beyond the tile), an unmasked vdivpd and a masked scatter, so 1 / 0
    //  raised the flag for a division by zero (T0359); a call per element
    //  divides only the diagonal
    static ULMBLAS_NOINLINE T
    reciprocal(const T &a)
    {
	return T(1) / a;
    }

    //  the mr x nr tile X (row major, rows of NR elements) to B; rows of B
    //  contiguous: row by row (contiguous stores)
    template <typename IndexType, typename TB>
    static void
    storeTile(IndexType mr, IndexType nr, const T *X, TB *B, IndexType incRowB,
              IndexType incColB)
    {
	if (incColB == IndexType(1)) {
	    for (IndexType r = 0; r < mr; ++r) {
		for (IndexType c = 0; c < nr; ++c) {
		    B[r * incRowB + c] = TB(X[r * NR + c]);
		}
	    }
	} else {
	    for (IndexType c = 0; c < nr; ++c) {
		for (IndexType r = 0; r < mr; ++r) {
		    B[r * incRowB + c * incColB] = TB(X[r * NR + c]);
		}
	    }
	}
    }

    //  kb x nc rows of B packed (PackB layout: panels of NR columns, row
    //  major); rows of B contiguous (A on the right, mapped to the left
    //  with transposed B): row by row, contiguous reads instead of NR
    //  elements of each of kb rows per panel
    template <typename IndexType, typename TB>
    static void
    packRowsB(IndexType kb, IndexType nc, const TB *B, IndexType incRowB,
              IndexType incColB, T *B_)
    {
	if (incColB != IndexType(1)) {
	    PackB<Backend>::run(kb, nc, false, B, incRowB, incColB, B_);
	    return;
	}
	const IndexType np = nc / NR, nr = nc % NR;
	for (IndexType r = 0; r < kb; ++r) {
	    const TB *b = &B[r * incRowB];
	    for (IndexType p = 0; p < np; ++p) {
		T *x = &B_[p * kb * NR + r * NR];
		for (IndexType c = 0; c < NR; ++c) {
		    x[c] = T(b[p * NR + c]);
		}
	    }
	    if (nr > 0) {
		T *x = &B_[np * kb * NR + r * NR];
		for (IndexType c = 0; c < NR; ++c) {
		    x[c] = c < nr ? T(b[np * NR + c]) : T(0);
		}
	    }
	}
    }

    //  the variant of the solution of a full diagonal tile (MR x MR):
    //
    //      substitution   with the prepared tile (inverted diagonal), row
    //                     by row
    //      inverse        the tile inverted once (per diagonal block), the
    //                     solution X = D^{-1} T by the micro kernel
    //
    //  The inverse pays off for tall tiles: on an Ice Lake core with the
    //  32 x 6 kernel dtrsm 256 x 8000 (getrf) 57 GFLOPS against 39, 8000 x
    //  256 (potrf) 49 against 36; on 64 threads +3 %.  Neutral for single
    //  precision; not for NEON (6 x 8) and SME (32 x 16) on an Apple M5.
    //  The variant: tuned().trsmInvert (ulmblas/config/tuned.h, measured by
    //  make tune; by default the inverse for MR >= 16 on Ice Lake),
    //  ULMBLAS_TRSM_INV=0|1 overrides it.
    static bool
    invertTiles()
    {
	static const std::ptrdiff_t env = [] {
	    const char *v = std::getenv("ULMBLAS_TRSM_INV");
	    return v && (v[0] == '0' || v[0] == '1')
	               ? std::ptrdiff_t(v[0] - '0')
		       : std::ptrdiff_t(-1);
	}();
	const std::ptrdiff_t t =
	    env >= 0 ? env : tuned().trsmInvert[TunedIndex<T>::value];
	if (t >= 0) {
	    return t == 1;
	}
	return MR >= 16 && platform() == Platform::IntelIceLake;
    }
};

//
//  The diagonal tiles of the packed kb x kb block A_ (panels of MR rows,
//  column major, panel p at A_ + p MR kb): diagonal inverted, zeros on the
//  other side of the diagonal (rows of the tile beyond the block are
//  padding, not read).
//
template <typename Frame>
template <typename IndexType>
void
TrsmFrame<Frame>::prepareDiagonal(bool lower, bool unitDiag, IndexType kb,
                                  IndexType p0, IndexType p1, T *A_, T *Inv)
{
    for (IndexType p = p0; p < p1; ++p) {
	const IndexType mr = std::min(IndexType(MR), IndexType(kb - p * MR));
	T *tile = &A_[p * MR * kb + p * MR * MR];
	//  zeros on the other side of the diagonal, then the diagonal: in
	//  loops of their own (a vectorized loop over a column could divide
	//  all its elements and raise the flag for a division by zero, T0359)
	for (IndexType q = 0; q < mr; ++q) {
	    const IndexType r0 = lower ? 0 : q + 1;
	    const IndexType r1 = lower ? q : mr;
	    for (IndexType r = r0; r < r1; ++r) {
		tile[q * MR + r] = T(0);
	    }
	}
	//  unitDiag outside the loop: a select in the loop lets the compiler
	//  divide also for a unit diagonal (the stored diagonal may be zero),
	//  e.g. clang with AVX-512 (T0359)
	if (unitDiag) {
	    for (IndexType q = 0; q < mr; ++q) {
		tile[q * MR + q] = T(1);
	    }
	} else {
	    for (IndexType q = 0; q < mr; ++q) {
		T &a = tile[q * MR + q];
		a = reciprocal(a);
	    }
	}

	//  the inverse of a full tile (Inv: column major, MR x MR per tile,
	//  the layout of a packed panel of A with k = MR): column q by
	//  substitution with the prepared tile
	if (Inv && mr == MR) {
	    T *Y = &Inv[p * MR * MR];
	    for (IndexType q = 0; q < MR; ++q) {
		T *y = &Y[q * MR];
		for (IndexType r = 0; r < MR; ++r) {
		    y[r] = T(0);
		}
		y[q] = tile[q * MR + q];
		for (IndexType t = 1; t < MR; ++t) {
		    const IndexType r = lower ? q + t : q - t;
		    if (r < 0 || r >= MR) {
			break;
		    }
		    T sum = T(0);
		    const IndexType i0 = lower ? q : r + 1;
		    const IndexType i1 = lower ? r : q + 1;
		    for (IndexType i = i0; i < i1; ++i) {
			sum += tile[i * MR + r] * y[i];
		    }
		    y[r] = -sum * tile[r * MR + r];
		}
	    }
	}
    }
}

//
//  The diagonal block: the kb rows of B_ (packed, panels of NR columns, row
//  major, panel j at B_ + j kb NR) solved in place with the prepared block
//  A_, the solution also stored in B (the kb x nc block).
//
template <typename Frame>
template <typename IndexType, typename TB>
void
TrsmFrame<Frame>::solveBlock(bool lower, IndexType kb, IndexType nc,
                             const T *A_, const T *Inv, T *B_, TB *B,
                             IndexType incRowB, IndexType incColB)
{
    const IndexType mp = (kb + MR - 1) / MR;
    const IndexType np = (nc + NR - 1) / NR;

    for (IndexType j = 0; j < np; ++j) {
	const IndexType nr = std::min(IndexType(NR), IndexType(nc - j * NR));
	T *Bj = &B_[j * kb * NR];

	for (IndexType t = 0; t < mp; ++t) {
	    const IndexType p = lower ? t : mp - 1 - t;
	    const IndexType r0 = p * MR;
	    const IndexType mr = std::min(IndexType(MR), kb - r0);
	    const T *Ap = &A_[p * MR * kb];
	    T *X = &Bj[r0 * NR]; // the tile in B_: row r at X + r NR

	    //  minus the product with the solved rows of the block: lower,
	    //  rows 0 ... r0-1 (columns 0 ... r0-1 of the panel), upper, rows
	    //  r0+mr ... kb-1.  The micro kernel computes the product into a
	    //  local column-major tile (the fast case of all kernels; zero
	    //  first, beta = 1: no kernel reads undefined values).
	    const IndexType k0 = lower ? 0 : r0 + mr;
	    const IndexType kk = lower ? r0 : kb - r0 - mr;
	    if (kk > 0) {
		alignas(64) T P[MR * NR] = {};
		Backend::ugemm(kk, T(1), &Ap[k0 * MR], &Bj[k0 * NR], T(1), P,
		               IndexType(1), IndexType(MR), Ap, Bj);
		for (IndexType r = 0; r < mr; ++r) {
		    for (IndexType c = 0; c < NR; ++c) {
			X[r * NR + c] -= P[c * MR + r];
		    }
		}
	    }

	    //  with the inverse of the tile (full tiles, invertTiles): X =
	    //  D^{-1} T by the micro kernel (T: MR rows of the panel of B_, the
	    //  layout of a packed panel of B with k = MR)
	    if (Inv && mr == MR) {
		alignas(64) T Q[MR * NR] = {};
		Backend::ugemm(IndexType(MR), T(1), &Inv[p * MR * MR], X, T(1),
		               Q, IndexType(1), IndexType(MR), Ap, Bj);
		for (IndexType r = 0; r < MR; ++r) {
		    for (IndexType c = 0; c < NR; ++c) {
			X[r * NR + c] = Q[c * MR + r];
		    }
		}
		storeTile(IndexType(MR), nr, X,
		          &B[r0 * incRowB + j * NR * incColB], incRowB,
		          incColB);
		continue;
	    }

	    //  substitution with the diagonal tile (inverted diagonal), row
	    //  by row; a row of the tile (NR elements) is contiguous
	    const T *D = &Ap[r0 * MR]; // element (r, q) at D[q MR + r]
	    for (IndexType t2 = 0; t2 < mr; ++t2) {
		const IndexType r = lower ? t2 : mr - 1 - t2;
		T *x = &X[r * NR];
		const IndexType q0 = lower ? 0 : r + 1;
		const IndexType q1 = lower ? r : mr;
		for (IndexType q = q0; q < q1; ++q) {
		    const T d = D[q * MR + r];
		    const T *y = &X[q * NR];
		    for (IndexType c = 0; c < NR; ++c) {
			x[c] -= d * y[c];
		    }
		}
		const T d = D[r * MR + r];
		for (IndexType c = 0; c < NR; ++c) {
		    x[c] *= d;
		}
	    }
	    storeTile(mr, nr, X, &B[r0 * incRowB + j * NR * incColB], incRowB,
	              incColB);
	}
    }
}

template <typename Frame>
template <typename IndexType, typename TA, typename TB>
void
TrsmFrame<Frame>::runSerial(bool lower, IndexType m, IndexType n, bool conjA,
                            bool unitDiag, const TA *A, IndexType incRowA,
                            IndexType incColA, TB *B, IndexType incRowB,
                            IndexType incColB)
{
    const std::ptrdiff_t KB = diagonalBlock(std::ptrdiff_t(m));
    static MemoryPool<T> memoryPool;

    const IndexType mcMax = std::max(IndexType(Blocks::MCmax), IndexType(KB));
    T *A_ = memoryPool.allocate(std::size_t(mcMax) * KB + MR);
    T *B_ = memoryPool.allocate(std::size_t(KB) * Blocks::NCmax + NR);
    T *Inv = invertTiles() ? memoryPool.allocate(std::size_t(KB) * MR + MR)
                           : nullptr;

    IndexType nc, mc;
    for (IndexType j = 0; j < n; j += nc) {
	nc = Blocks::blockSize(n - j, Blocks::NC, Blocks::NCmax);

	//  the diagonal blocks: lower from the top, upper from the bottom
	//  (the last block of the upper case first, its size the rest)
	const IndexType nb = (m + KB - 1) / KB;
	for (IndexType b = 0; b < nb; ++b) {
	    const IndexType l = lower ? b * KB : (nb - 1 - b) * KB;
	    const IndexType kb = std::min(IndexType(KB), m - l);

	    TB *Bl = &B[l * incRowB + j * incColB];
	    PackA<Backend>::run(kb, kb, conjA, &A[l * (incRowA + incColA)],
	                        incRowA, incColA, A_);
	    prepareDiagonal(lower, unitDiag, kb, IndexType(0),
	                    IndexType((kb + MR - 1) / MR), A_, Inv);
	    //  B packed and solved CHUNK panels at a time, while the panels are
	    //  in the cache (as OpenBLAS's trsm_L.c with 3 UNROLL_N columns);
	    //  packing all nc columns first lets them leave the L2 cache: on
	    //  heim (i5-12500, 6 threads) dtrsm 256 x 8000 146 -> 185 GFLOPS,
	    //  8000 x 256 (A on the right) 121 -> 185 (with packRowsB)
	    constexpr IndexType CHUNK = 2 * NR;
	    for (IndexType jj = 0; jj < nc; jj += CHUNK) {
		const IndexType cc = std::min(CHUNK, nc - jj);
		packRowsB(kb, cc, &Bl[jj * incColB], incRowB, incColB,
		          &B_[jj * kb]);
		solveBlock(lower, kb, cc, A_, Inv, &B_[jj * kb],
		           &Bl[jj * incColB], incRowB, incColB);
	    }

	    //  the rows below (above): C <- C - A(rows, l:l+kb) X
	    const IndexType i0 = lower ? l + kb : IndexType(0);
	    const IndexType i1 = lower ? m : l;
	    for (IndexType i = i0; i < i1; i += mc) {
		mc = Blocks::blockSize(i1 - i, Blocks::MC, Blocks::MCmax);
		PackA<Backend>::run(mc, kb, conjA,
		                    &A[i * incRowA + l * incColA], incRowA,
		                    incColA, A_);
		Macro::run(mc, nc, kb, T(-1), A_, B_, T(1),
		           &B[i * incRowB + j * incColB], incRowB, incColB);
	    }
	}
    }

    if (Inv) {
	memoryPool.release(Inv);
    }
    memoryPool.release(A_);
    memoryPool.release(B_);
}

//
//  Threads that share the packed blocks (TrShared, shared): for a block of
//  nc columns of B, per diagonal block (rows l ... l+kb-1)
//
//      - the threads pack the diagonal block of A (some of its panels each)
//        into a common buffer and prepare its diagonal tiles; barrier;
//      - each thread solves its panels of the nc columns, packed into the
//        common buffer B_ of the solution; barrier;
//      - the rows below (above) are updated in rectangles (rows and
//        columns, whole panels, as the threads of gemm), each thread packs
//        the block of A of its rows; then its panels of the next diagonal
//        block.
//
//  The barrier before a solution also orders the updates before it (the
//  rows of the next diagonal block are updated by all threads); the one
//  after it orders the packing of the next diagonal block (and of the next
//  solution) after the last reads of the buffers.
//
template <typename Frame>
template <typename IndexType, typename TA, typename TB>
void
TrsmFrame<Frame>::runShared(std::ptrdiff_t nt, bool lower, IndexType m,
                            IndexType n, bool conjA, bool unitDiag, const TA *A,
                            IndexType incRowA, IndexType incColA, TB *B,
                            IndexType incRowB, IndexType incColB)
{
    const std::ptrdiff_t KB = diagonalBlock(std::ptrdiff_t(m));
    static MemoryPool<T> memoryPool;

    const std::size_t sizeA = std::size_t(Blocks::MCmax) * KB + MR;
    T *Ad = memoryPool.allocate(std::size_t(KB) * KB + MR);
    T *Inv = invertTiles() ? memoryPool.allocate(std::size_t(KB) * MR + MR)
                           : nullptr;
    T *B_ = memoryPool.allocate(std::size_t(KB) * Blocks::NCmax + NR);
    T *A_ = memoryPool.allocate(sizeA * std::size_t(nt));
    TreeBarrier barrier(nt);

    const IndexType nb = (m + KB - 1) / KB;

    auto body = [&](std::ptrdiff_t tid, std::ptrdiff_t threads) {
	//  fewer threads than planned (pool in use): thread 0 alone
	if (threads != nt) {
	    if (tid == 0) {
		runSerial(lower, m, n, conjA, unitDiag, A, incRowA, incColA, B,
		          incRowB, incColB);
	    }
	    return;
	}
	T *Au = A_ + sizeA * std::size_t(tid);

	//  this thread's panels of diagonal block b (lower from the top,
	//  upper from the bottom), packed and prepared
	auto packDiagonal = [&](IndexType b) {
	    const IndexType l = lower ? b * KB : (nb - 1 - b) * KB;
	    const IndexType kb = std::min(IndexType(KB), m - l);
	    const IndexType mp = (kb + MR - 1) / MR;
	    const IndexType p0 = IndexType(tid) * mp / IndexType(nt);
	    const IndexType p1 = IndexType(tid + 1) * mp / IndexType(nt);
	    if (p1 > p0) {
		const IndexType r0 = p0 * MR;
		const IndexType r1 = std::min(kb, IndexType(p1 * MR));
		PackA<Backend>::run(r1 - r0, kb, conjA,
		                    &A[(l + r0) * incRowA + l * incColA],
		                    incRowA, incColA, &Ad[r0 * kb]);
		prepareDiagonal(lower, unitDiag, kb, p0, p1, Ad, Inv);
	    }
	};

	packDiagonal(IndexType(0));
	IndexType nc;
	for (IndexType j = 0; j < n; j += nc) {
	    nc = Blocks::blockSize(n - j, Blocks::NC, Blocks::NCmax);

	    //  this thread's panels of the nc columns in the solution
	    const IndexType np = (nc + NR - 1) / NR;
	    const IndexType c0 = std::min(
	        nc, IndexType(IndexType(tid) * np / IndexType(nt) * NR));
	    const IndexType c1 = std::min(
	        nc, IndexType(IndexType(tid + 1) * np / IndexType(nt) * NR));

	    for (IndexType b = 0; b < nb; ++b) {
		const IndexType l = lower ? b * KB : (nb - 1 - b) * KB;
		const IndexType kb = std::min(IndexType(KB), m - l);
		TB *Bl = &B[l * incRowB + j * incColB];

		barrier.wait(tid);
		constexpr IndexType CHUNK = 2 * NR;
		for (IndexType jj = c0; jj < c1; jj += CHUNK) {
		    const IndexType cc = std::min(CHUNK, c1 - jj);
		    packRowsB(kb, cc, &Bl[jj * incColB], incRowB, incColB,
		              &B_[jj * kb]);
		    solveBlock(lower, kb, cc, Ad, Inv, &B_[jj * kb],
		               &Bl[jj * incColB], incRowB, incColB);
		}
		barrier.wait(tid);

		const IndexType i0 = lower ? l + kb : IndexType(0);
		const IndexType i1 = lower ? m : l;
		TrShared::update<Frame>(tid, nt, i0, i1, nc, kb, T(-1), conjA,
		                        &A[l * incColA], incRowA, incColA, B_,
		                        &B[j * incColB], incRowB, incColB, Au);

		//  the next diagonal block (of these or the next columns)
		if (b + 1 < nb) {
		    packDiagonal(b + 1);
		} else if (j + nc < n) {
		    packDiagonal(IndexType(0));
		}
	    }
	}
    };
    ThreadPool::instance().run(nt, body);

    memoryPool.release(A_);
    memoryPool.release(B_);
    if (Inv) {
	memoryPool.release(Inv);
    }
    memoryPool.release(Ad);
}

template <typename Frame>
template <typename IndexType, typename Alpha, typename TA, typename TB>
void
TrsmFrame<Frame>::run(bool lower, IndexType m, IndexType n, const Alpha &alpha,
                      bool conjA, bool unitDiag, const TA *A, IndexType incRowA,
                      IndexType incColA, TB *B, IndexType incRowB,
                      IndexType incColB)
{
    const std::ptrdiff_t KB = diagonalBlock(std::ptrdiff_t(m));
    if (m <= 0 || n <= 0) {
	return;
    }
    //  alpha first (as OpenBLAS's GEMM_BETA)
    if (alpha != Alpha(1)) {
	for (IndexType j = 0; j < n; ++j) {
	    for (IndexType i = 0; i < m; ++i) {
		TB &b = B[i * incRowB + j * incColB];
		b = alpha == Alpha(0) ? TB(0) : TB(alpha * b);
	    }
	}
	if (alpha == Alpha(0)) {
	    return;
	}
    }

    //  threads: the work m^2 n / 2; the variant (TrShared)
    std::ptrdiff_t nt = GemmThreads::forProblem<T>(m, n, (m + 1) / 2);
    const std::ptrdiff_t np = (std::ptrdiff_t(n) + NR - 1) / NR;
    if (nt > 1 && TrShared::useShared("ULMBLAS_TRSM_SHARED", tuned().trsm,
                                      TunedIndex<T>::value, nt,
                                      std::ptrdiff_t(m), KB, np)) {
	runShared(nt, lower, m, n, conjA, unitDiag, A, incRowA, incColA, B,
	          incRowB, incColB);
	return;
    }
    //  columns: at least minPanels panels (NR columns) per thread
    const std::ptrdiff_t minPanels = TrShared::minPanels(
        "ULMBLAS_TRSM_MINP", tuned().trsm, TunedIndex<T>::value);
    nt = std::min(nt, std::max(std::ptrdiff_t(1), np / minPanels));
    if (nt <= 1) {
	runSerial(lower, m, n, conjA, unitDiag, A, incRowA, incColA, B, incRowB,
	          incColB);
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
	    runSerial(lower, m, j1 - j0, conjA, unitDiag, A, incRowA, incColA,
	              &B[j0 * incColB], incRowB, incColB);
	}
    };
    ThreadPool::instance().run(nt, body);
}

//
//  trsm in the selection tables (gemm/plan.h): the frame of trsm with the
//  frame of the entry (for a kernel for small matrices the frame it falls
//  back to).  The problem is looked up as an m x n x m product.
//
template <typename IndexType, typename Alpha, typename TA, typename TB>
struct TrsmPlan
{
    typedef void (*Run)(bool lower, IndexType m, IndexType n,
                        const Alpha &alpha, bool conjA, bool unitDiag,
                        const TA *A, IndexType incRowA, IndexType incColA,
                        TB *B, IndexType incRowB, IndexType incColB);

    Run run;
    const char *name;
};

template <typename IndexType, typename Alpha, typename TA, typename TB>
struct TrsmOp
{
    typedef TrsmPlan<IndexType, Alpha, TA, TB> Plan;

    static constexpr std::size_t sizeC = sizeof(TB);
    static constexpr std::size_t sizeA = sizeof(TA);

    template <typename F>
    static Plan
    plan(const char *name)
    {
	typedef TrsmFrame<typename GemmFrameOf<F>::type> Frame;
	Plan p = {&Frame::template run<IndexType, Alpha, TA, TB>, name};
	return p;
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_FRAMETRSM_H
