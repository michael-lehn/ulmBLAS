//
//  DPOTRF: tiled Cholesky factorization with OpenMP task dependencies.
//
//  Right-looking tile algorithm (as PLASMA's pdpotrf), in place on the
//  column-major matrix.  The tiles are nb x nb blocks of A; a byte per tile
//  (tok) stands for the tile in the depend clauses:
//
//      for k = 0, ..., nt-1
//          POTRF  A(k,k) = L(k,k) L(k,k)^T             inout A(k,k)
//          for i > k
//              TRSM   A(i,k) = A(i,k) L(k,k)^{-T}      in A(k,k), inout A(i,k)
//          for i > k
//              SYRK   A(i,i) -= A(i,k) A(i,k)^T        in A(i,k), inout A(i,i)
//              for k < j < i
//                  GEMM  A(i,j) -= A(i,k) A(j,k)^T     in A(i,k), A(j,k),
//                                                      inout A(i,j)
//
//  The tasks on the critical path (POTRF, the TRSMs of column k and the
//  updates of column k+1) have higher priorities: the factorization of the
//  next column can start while the rest of the trailing matrix is updated
//  (lookahead by priority, as in PLASMA).
//
//  The upper variant A = U^T U is the lower variant of the transposed view
//  (row and column stride exchanged): U = L^T.
//

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <limits>
#include <vector>

#include <omp.h>

#include <ulmlapack.h>
#include <ulmlapack.hpp>
#include "common.h"
#include "kernels.h"
#include "tiles.h"
#include "checkdeps.h"

namespace ulmlapack {

namespace {

const int PRIO_POTRF = 3;
const int PRIO_CRIT = 2;
const int PRIO_NEXT = 1;
const int PRIO_OTHER = 0;

//
//  W: the tiles the graph works on (in place, a column-major copy or the
//  tile layout).  src: the caller's matrix if W is a copy; then the graph
//  starts with a copy task per tile (lower triangle) and ends with one that
//  copies the tile back, as soon as its last update is done.
//
//  prepack: every tile L(i,k) below the diagonal is packed once, at the end
//  of its TRSM, into the two formats of ulmBLAS's GEMM micro kernel: as
//  the left operand (row panels of MR, gepack_A) and, transposed, as the
//  right one (column panels of NR, gepack_B).  The GEMM tasks then call the
//  macro kernel mgemm directly instead of ulmBLAS::gemm, which would pack
//  both operands again in every task.  The packed tiles belong to the
//  token of their tile: written by the TRSM, read by tasks that depend on
//  it.  SYRK uses them too if the diagonal tile may be written completely
//  (a copy: the upper triangle is never copied back); in place and on a
//  TileMatrix (the caller's data) it keeps sylrk.
//
Index
potrf_tasks(Index n, Tiles W, const Source *src, Index nb, int threads,
            bool empty, bool prepack = false)
{
    const Index nt = (n + nb - 1) / nb;
    const Index rs = W.rs, cs = W.cs;

    typedef ulmBLAS::BlockSize<double> BS;
    typedef ulmBLAS::BlockSizeUGemm<double> BU;
    const Index MR = BU::MR, NR = BU::NR;
    // a tile has to fit into one block of the macro kernel
    prepack = prepack && nb <= Index(BS::MC) && nb <= Index(BS::NC) &&
              nb <= Index(BS::KC);
    const Index sizeA = (nb + MR - 1) / MR * MR * nb + MR;
    const Index sizeB = (nb + NR - 1) / NR * NR * nb + NR;
    double *packed =
        prepack ? workspace(std::size_t(nt) * nt * (sizeA + sizeB), 1) : 0;
    auto packA = [=](Index i, Index k) {
	return packed + (i * nt + k) * (sizeA + sizeB);
    };
    auto packB = [=](Index i, Index k) { return packA(i, k) + sizeA; };
    const bool fullDiag = src != 0;

    std::vector<char> tok(nt * nt);
    char *T = tok.data();
    (void)T; // only used in depend clauses (GCC warns)

    // step of the first failed POTRF (max: none) and its INFO
    std::atomic<Index> failStep(std::numeric_limits<Index>::max());
    // empty tasks: every body is skipped as after a failure at step -1
    const Index skipFrom = empty ? -1 : std::numeric_limits<Index>::max();
    Index info = 0;

    CHECKDEPS_LOG(log, std::size_t(nt) * nt * nt + 5 * nt * nt + nt);

    CHECKDEPS_TIME(tStart);
    CHECKDEPS_TIME(tGenEnd);
#pragma omp parallel num_threads(threads) shared(failStep, info)
#pragma omp single
    {
	if (src) {
	    for (Index j = 0; j < nt; ++j) {
		for (Index i = j; i < nt; ++i) {
		    const Index mi = std::min(nb, n - i * nb);
		    const Index nj = std::min(nb, n - j * nb);
		    double *Wij = W.tile(i, j);
		    const double *Sij =
		        &src->A[i * nb * src->rs + j * nb * src->cs];

		    CHECKDEPS_CREATE(rc, log, "COPYIN", -1, i, j);
		    CHECKDEPS_ACCESS(rc, i * nt + j, true);
#pragma omp task depend(out : T[i * nt + j])                                   \
    priority(j <= 1 ? PRIO_CRIT : PRIO_OTHER)
		    {
			taskEnter();
			CHECKDEPS_START(rc);
			if (!empty) {
			    copyBlock(mi, nj, Sij, src->rs, src->cs, Wij, rs,
			              cs, i == j);
			}
			CHECKDEPS_END(rc);
			taskLeave();
		    }
		}
	    }
	}
	for (Index k = 0; k < nt; ++k) {
	    const Index nk = std::min(nb, n - k * nb);
	    double *Akk = W.tile(k, k);

	    CHECKDEPS_CREATE(rp, log, "POTRF", k, k, k);
	    CHECKDEPS_ACCESS(rp, k * nt + k, true);
#pragma omp task depend(inout : T[k * nt + k]) priority(PRIO_POTRF)
	    {
		taskEnter();
		CHECKDEPS_START(rp);
		if (std::min(skipFrom,
		             failStep.load(std::memory_order_relaxed)) > k) {
		    Index inf = potrf_lower_packed(nk, Akk, rs, cs);
		    if (inf) {
			// tasks of later steps depend on this one, so the
			// first failure is the one with the smallest k
			info = k * nb + inf;
			failStep.store(k);
		    }
		}
		CHECKDEPS_END(rp);
		taskLeave();
	    }
	    for (Index i = k + 1; i < nt; ++i) {
		const Index ni = std::min(nb, n - i * nb);
		double *Aik = W.tile(i, k);

		CHECKDEPS_CREATE(rt, log, "TRSM", k, i, k);
		CHECKDEPS_ACCESS(rt, k * nt + k, false);
		CHECKDEPS_ACCESS(rt, i * nt + k, true);
#pragma omp task depend(in : T[k * nt + k]) depend(inout : T[i * nt + k])      \
    priority(i == k + 1 ? PRIO_POTRF : PRIO_CRIT)
		{
		    taskEnter();
		    CHECKDEPS_START(rt);
		    if (std::min(skipFrom,
		                 failStep.load(std::memory_order_relaxed)) >
		        k) {
			// A(i,k) L(k,k)^T = A(i,k)
			ulmBLAS::trlsm(nk, ni, 1.0, false, false, Akk, rs, cs,
			               Aik, cs, rs);
			if (prepack) {
			    ulmBLAS::gepack_A(ni, nk, false, Aik, rs, cs,
			                      packA(i, k));
			    ulmBLAS::gepack_B(nk, ni, false, Aik, cs, rs,
			                      packB(i, k));
			}
		    }
		    CHECKDEPS_END(rt);
		    taskLeave();
		}
	    }
	    for (Index i = k + 1; i < nt; ++i) {
		const Index ni = std::min(nb, n - i * nb);
		const double *Aik = W.tile(i, k);
		double *Aii = W.tile(i, i);

		CHECKDEPS_CREATE(rs_, log, "SYRK", k, i, i);
		CHECKDEPS_ACCESS(rs_, i * nt + k, false);
		CHECKDEPS_ACCESS(rs_, i * nt + i, true);
#pragma omp task depend(in : T[i * nt + k]) depend(inout : T[i * nt + i])      \
    priority(i == k + 1 ? PRIO_POTRF : PRIO_NEXT)
		{
		    taskEnter();
		    CHECKDEPS_START(rs_);
		    if (std::min(skipFrom,
		                 failStep.load(std::memory_order_relaxed)) >
		        k) {
			if (prepack && fullDiag) {
			    ulmBLAS::mgemm(ni, ni, nk, -1.0, packA(i, k),
			                   packB(i, k), 1.0, Aii, rs, cs);
			} else {
			    ulmBLAS::sylrk(ni, nk, -1.0, Aik, rs, cs, 1.0, Aii,
			                   rs, cs);
			}
		    }
		    CHECKDEPS_END(rs_);
		    taskLeave();
		}
		for (Index j = k + 1; j < i; ++j) {
		    const Index nj = std::min(nb, n - j * nb);
		    const double *Ajk = W.tile(j, k);
		    double *Aij = W.tile(i, j);

		    CHECKDEPS_CREATE(rg, log, "GEMM", k, i, j);
		    CHECKDEPS_ACCESS(rg, i * nt + k, false);
		    CHECKDEPS_ACCESS(rg, j * nt + k, false);
		    CHECKDEPS_ACCESS(rg, i * nt + j, true);
#pragma omp task depend(in : T[i * nt + k], T[j * nt + k])                     \
    depend(inout : T[i * nt + j])                                              \
    priority(j == k + 1 ? PRIO_CRIT : PRIO_OTHER)
		    {
			taskEnter();
			CHECKDEPS_START(rg);
			if (std::min(skipFrom,
			             failStep.load(std::memory_order_relaxed)) >
			    k) {
			    if (prepack) {
				ulmBLAS::mgemm(ni, nj, nk, -1.0, packA(i, k),
				               packB(j, k), 1.0, Aij, rs, cs);
			    } else {
				ulmBLAS::gemm(ni, nj, nk, -1.0, false, Aik, rs,
				              cs, false, Ajk, cs, rs, 1.0, Aij,
				              rs, cs);
			    }
			}
			CHECKDEPS_END(rg);
			taskLeave();
		    }
		}
	    }
	}
	if (src) {
	    for (Index j = 0; j < nt; ++j) {
		for (Index i = j; i < nt; ++i) {
		    const Index mi = std::min(nb, n - i * nb);
		    const Index nj = std::min(nb, n - j * nb);
		    const double *Wij = W.tile(i, j);
		    double *Sij = &src->A[i * nb * src->rs + j * nb * src->cs];

		    CHECKDEPS_CREATE(rb, log, "COPYBACK", nt, i, j);
		    CHECKDEPS_ACCESS(rb, i * nt + j, false);
#pragma omp task depend(in : T[i * nt + j]) priority(PRIO_OTHER)
		    {
			taskEnter();
			CHECKDEPS_START(rb);
			if (!empty) {
			    copyBlock(mi, nj, Wij, rs, cs, Sij, src->rs,
			              src->cs, i == j);
			}
			CHECKDEPS_END(rb);
			taskLeave();
		    }
		}
	    }
	}
	CHECKDEPS_TIME_SET(tGenEnd);
    }
    CHECKDEPS_TIME(tEnd);
    CHECKDEPS_PROFILE(log, threads, tStart, tGenEnd, tEnd);
    if (CHECKDEPS_CHECK(log, "dpotrf")) {
	std::abort();
    }
    return info;
}

//
//  Outer blocking for large strides (param outer = s > 1), in place.
//
//  The right-looking tile algorithm reads and writes every tile of the
//  trailing matrix once per tile step: about n/nb passes over the strided
//  matrix, each with k = nb.  With a large row stride every element is its
//  own cache line, so these passes cost most of the time.  Here the tile
//  columns are grouped into outer blocks of s tiles:
//
//    - within an outer block (all rows, s tile columns): the tile
//      algorithm, POTRF, TRSM, SYRK, GEMM as in potrf_tasks.  Every TRSM
//      packs its tile L(i,k) once in the two formats of the GEMM micro
//      kernel (as with prepack); the GEMMs call the macro kernel mgemm.
//    - the trailing matrix: one UPDATE task per tile (i,j) applies the
//      pending blocks, sum of L(i,k) L(j,k)^T, accumulated with mgemm in a
//      contiguous buffer (in the cache), then A(i,j) -= C (lower triangle
//      for i == j): the strided tile is read and written once per update.
//
//  window w: the updates of a trailing tile are deferred until its column
//  belongs to the next outer block or w blocks are pending; then all of
//  them are applied at once.  w times fewer passes over the trailing
//  matrix, for w+1 outer blocks of packed tiles.
//
//  strip: the tiles of the current outer block are copied (one task per
//  tile) into contiguous nb x nb tiles, the block is factored there (no
//  strides: POTRF, TRSM, packing, updates within the block) and every tile
//  is copied back once it is final.  A strided tile is read and written
//  once while its block is processed instead of in every task.
//
//  lean (strip = 2): no packed tiles.  The block tiles are contiguous
//  anyway (strip), so the TRSMs don't pack and the GEMMs and UPDATEs call
//  ulmBLAS::gemm on the contiguous tiles (it packs from the cache, not
//  from the strided matrix).  The strip tiles then have to live until the
//  last UPDATE that reads them: a ring of w+1 outer blocks.
//
//  Memory: packed tiles of w+1 outer blocks, about 2 (w+1) n s nb doubles
//  (tokens P), with strip the contiguous tiles of two outer blocks, about
//  2 n s nb doubles (tokens Q), and one nb x nb buffer per thread; no copy
//  of the matrix.  lean: only the strip tiles, (w+1) n s nb doubles.  The
//  rings: a slot is overwritten only after all readers of its previous use are
//  done (they depend on its token with 'in').
//
Index
potrf_outer_tasks(Index n, Tiles A, Index nb, Index s, Index w, int stripMode,
                  int threads, bool empty)
{
    const bool strip = stripMode > 0;
    const bool lean = stripMode == 2;
    const Index nt = (n + nb - 1) / nb;
    const Index rs = A.rs, cs = A.cs;
    const Index ring = lean ? 1 : (w + 1) * s;  // packed slots per tile row
    const Index sring = (lean ? w + 1 : 2) * s; // strip slots per tile row

    typedef ulmBLAS::BlockSizeUGemm<double> BU;
    const Index MR = BU::MR, NR = BU::NR;
    const Index sizeA = (nb + MR - 1) / MR * MR * nb + MR;
    const Index sizeB = (nb + NR - 1) / NR * NR * nb + NR;
    double *packed =
        lean ? 0 : workspace(std::size_t(nt) * ring * (sizeA + sizeB), 1);
    double *sbuf = strip ? workspace(std::size_t(nt) * sring * nb * nb, 0) : 0;
    auto pslot = [=](Index i, Index k) { return i * ring + k % ring; };
    auto qslot = [=](Index i, Index k) { return i * sring + k % sring; };
    auto packA = [=](Index i, Index k) {
	return packed + pslot(i, k) * (sizeA + sizeB);
    };
    auto packB = [=](Index i, Index k) { return packA(i, k) + sizeA; };

    //  the tile (i,k) of the current outer block: contiguous (strip) or in
    //  place; strides of the block tiles
    auto btile = [=](Index i, Index k) {
	return strip ? sbuf + qslot(i, k) * nb * nb : A.tile(i, k);
    };
    const Index brs = strip ? Index(1) : rs;
    const Index bcs = strip ? nb : cs;

    std::vector<char> tok(nt * nt + nt * ring + nt * sring);
    char *T = tok.data();    // tiles
    char *P = T + nt * nt;   // packed slots
    char *Q = P + nt * ring; // strip slots
    (void)T;
    (void)P;
    (void)Q;

    std::vector<Index> flushed(nt, -1); // last block applied to column

    std::atomic<Index> failStep(std::numeric_limits<Index>::max());
    const Index skipFrom = empty ? -1 : std::numeric_limits<Index>::max();
    Index info = 0;
    auto run = [&](Index k) {
	return std::min(skipFrom, failStep.load(std::memory_order_relaxed)) > k;
    };

    CHECKDEPS_LOG(log, std::size_t(nt) * nt * nt + 7 * nt * nt + nt);

    CHECKDEPS_TIME(tStart);
    CHECKDEPS_TIME(tGenEnd);
#pragma omp parallel num_threads(threads) shared(failStep, info)
#pragma omp single
    {
	for (Index K0 = 0; K0 < nt; K0 += s) {
	    const Index K1 = std::min(nt, K0 + s);
	    const Index b = K0 / s;

	    //  strip: copy the tiles of the block in
	    if (strip) {
		for (Index j = K0; j < K1; ++j) {
		    for (Index i = j; i < nt; ++i) {
			const Index mi = std::min(nb, n - i * nb);
			const Index nj = std::min(nb, n - j * nb);
			const double *Aij = A.tile(i, j);
			double *Bij = btile(i, j);

			CHECKDEPS_CREATE(rc, log, "COPYIN", K0, i, j);
			CHECKDEPS_ACCESS(rc, i * nt + j, true);
			CHECKDEPS_ACCESS(rc, nt * nt + nt * ring + qslot(i, j),
			                 true);
#pragma omp task depend(inout : T[i * nt + j], Q[qslot(i, j)])                 \
    priority(j == K0 ? PRIO_POTRF : PRIO_CRIT)
			{
			    taskEnter();
			    CHECKDEPS_START(rc);
			    if (!empty) {
				copyBlock(mi, nj, Aij, rs, cs, Bij, 1, nb,
				          i == j);
			    }
			    CHECKDEPS_END(rc);
			    taskLeave();
			}
		    }
		}
	    }

	    //  the outer block: tile algorithm on tile columns K0..K1-1
	    for (Index k = K0; k < K1; ++k) {
		const Index nk = std::min(nb, n - k * nb);
		double *Bkk = btile(k, k);

		CHECKDEPS_CREATE(rp, log, "POTRF", k, k, k);
		CHECKDEPS_ACCESS(rp, k * nt + k, true);
		CHECKDEPS_ACCESS(rp, nt * nt + nt * ring + qslot(k, k), false);
#pragma omp task depend(inout : T[k * nt + k]) depend(in : Q[qslot(k, k)])     \
    priority(PRIO_POTRF)
		{
		    taskEnter();
		    CHECKDEPS_START(rp);
		    if (run(k)) {
			Index inf = potrf_lower_packed(nk, Bkk, brs, bcs);
			if (inf) {
			    info = k * nb + inf;
			    failStep.store(k);
			}
		    }
		    CHECKDEPS_END(rp);
		    taskLeave();
		}
		for (Index i = k + 1; i < nt; ++i) {
		    const Index ni = std::min(nb, n - i * nb);
		    double *Bik = btile(i, k);

		    CHECKDEPS_CREATE(rt, log, "TRSM", k, i, k);
		    CHECKDEPS_ACCESS(rt, k * nt + k, false);
		    CHECKDEPS_ACCESS(rt, i * nt + k, true);
		    CHECKDEPS_ACCESS(rt, nt * nt + pslot(i, k), true);
		    CHECKDEPS_ACCESS(rt, nt * nt + nt * ring + qslot(i, k),
		                     false);
		    CHECKDEPS_ACCESS(rt, nt * nt + nt * ring + qslot(k, k),
		                     false);
#pragma omp task depend(in : T[k * nt + k], Q[qslot(i, k)], Q[qslot(k, k)])    \
    depend(inout : T[i * nt + k], P[pslot(i, k)])                              \
    priority(i < K1 ? PRIO_POTRF : PRIO_CRIT)
		    {
			taskEnter();
			CHECKDEPS_START(rt);
			if (run(k)) {
			    ulmBLAS::trlsm(nk, ni, 1.0, false, false, Bkk, brs,
			                   bcs, Bik, bcs, brs);
			    if (!lean) {
				ulmBLAS::gepack_A(ni, nk, false, Bik, brs, bcs,
				                  packA(i, k));
				ulmBLAS::gepack_B(nk, ni, false, Bik, bcs, brs,
				                  packB(i, k));
			    }
			}
			CHECKDEPS_END(rt);
			taskLeave();
		    }
		}
		//  updates of the tile columns k+1..K1-1 of the block
		for (Index j = k + 1; j < K1; ++j) {
		    const Index nj = std::min(nb, n - j * nb);
		    for (Index i = j; i < nt; ++i) {
			const Index ni = std::min(nb, n - i * nb);
			double *Bij = btile(i, j);
			const double *Bik = btile(i, k);

			CHECKDEPS_CREATE(rg, log, i == j ? "SYRK" : "GEMM", k,
			                 i, j);
			CHECKDEPS_ACCESS(rg, i * nt + k, false);
			CHECKDEPS_ACCESS(rg, j * nt + k, false);
			CHECKDEPS_ACCESS(rg, nt * nt + pslot(i, k), false);
			CHECKDEPS_ACCESS(rg, nt * nt + pslot(j, k), false);
			CHECKDEPS_ACCESS(rg, nt * nt + nt * ring + qslot(i, k),
			                 false);
			CHECKDEPS_ACCESS(rg, nt * nt + nt * ring + qslot(j, k),
			                 false);
			CHECKDEPS_ACCESS(rg, nt * nt + nt * ring + qslot(i, j),
			                 false);
			CHECKDEPS_ACCESS(rg, i * nt + j, true);
#pragma omp task depend(in : T[i * nt + k], T[j * nt + k], P[pslot(i, k)],     \
                            P[pslot(j, k)], Q[qslot(i, k)], Q[qslot(j, k)],    \
                            Q[qslot(i, j)]) depend(inout : T[i * nt + j])      \
    priority(j == k + 1 ? PRIO_CRIT : PRIO_NEXT)
			{
			    taskEnter();
			    CHECKDEPS_START(rg);
			    if (run(k)) {
				if (i == j) {
				    ulmBLAS::sylrk(ni, nk, -1.0, Bik, brs, bcs,
				                   1.0, Bij, brs, bcs);
				} else if (lean) {
				    ulmBLAS::gemm(ni, nj, nk, -1.0, false, Bik,
				                  brs, bcs, false, btile(j, k),
				                  bcs, brs, 1.0, Bij, brs, bcs);
				} else {
				    ulmBLAS::mgemm(ni, nj, nk, -1.0,
				                   packA(i, k), packB(j, k),
				                   1.0, Bij, brs, bcs);
				}
			    }
			    CHECKDEPS_END(rg);
			    taskLeave();
			}
		    }
		}
	    }

	    //  strip: copy the final tiles of the block back
	    if (strip) {
		for (Index j = K0; j < K1; ++j) {
		    for (Index i = j; i < nt; ++i) {
			const Index mi = std::min(nb, n - i * nb);
			const Index nj = std::min(nb, n - j * nb);
			double *Aij = A.tile(i, j);
			const double *Bij = btile(i, j);

			CHECKDEPS_CREATE(rb, log, "COPYBACK", K0, i, j);
			CHECKDEPS_ACCESS(rb, i * nt + j, false);
			CHECKDEPS_ACCESS(rb, nt * nt + nt * ring + qslot(i, j),
			                 false);
#pragma omp task depend(in : T[i * nt + j], Q[qslot(i, j)]) priority(PRIO_OTHER)
			{
			    taskEnter();
			    CHECKDEPS_START(rb);
			    if (!empty) {
				copyBlock(mi, nj, Bij, 1, nb, Aij, rs, cs,
				          i == j);
			    }
			    CHECKDEPS_END(rb);
			    taskLeave();
			}
		    }
		}
	    }

	    //  trailing matrix: apply the pending blocks to a column if it
	    //  belongs to the next outer block or w blocks are pending
	    for (Index j = K1; j < nt; ++j) {
		const bool next = j < K1 + s;
		if (!next && b - flushed[j] < w) {
		    continue;
		}
		const Index Ks =
		    (flushed[j] + 1) * s; // first pending tile column
		flushed[j] = b;
		const Index nj = std::min(nb, n - j * nb);
		for (Index i = j; i < nt; ++i) {
		    const Index ni = std::min(nb, n - i * nb);
		    double *Aij = A.tile(i, j);

		    CHECKDEPS_CREATE(ru, log, "UPDATE", Ks, i, j);
		    for (Index kk = Ks; kk < K1; ++kk) {
			CHECKDEPS_ACCESS(ru, i * nt + kk, false);
			CHECKDEPS_ACCESS(ru, j * nt + kk, false);
			CHECKDEPS_ACCESS(ru, nt * nt + pslot(i, kk), false);
			CHECKDEPS_ACCESS(ru, nt * nt + pslot(j, kk), false);
			CHECKDEPS_ACCESS(ru, nt * nt + nt * ring + qslot(i, kk),
			                 false);
			CHECKDEPS_ACCESS(ru, nt * nt + nt * ring + qslot(j, kk),
			                 false);
		    }
		    CHECKDEPS_ACCESS(ru, i * nt + j, true);
#pragma omp task depend(                                                       \
        iterator(kk = Ks : K1), in : T[i * nt + kk], T[j * nt + kk],           \
            P[i * ring + kk % ring], P[j * ring + kk % ring],                  \
            Q[i * sring + kk % sring], Q[j * sring + kk % sring])              \
    depend(inout : T[i * nt + j]) priority(next ? PRIO_NEXT : PRIO_OTHER)
		    {
			taskEnter();
			CHECKDEPS_START(ru);
			if (run(K1 - 1)) {
			    thread_local std::vector<double> buf;
			    buf.assign(std::size_t(nb) * nb, 0.0);
			    double *C = buf.data();
			    for (Index kk = Ks; kk < K1; ++kk) {
				if (lean) {
				    ulmBLAS::gemm(
				        ni, nj, nb, 1.0, false, btile(i, kk),
				        Index(1), nb, false, btile(j, kk), nb,
				        Index(1), 1.0, C, Index(1), nb);
				} else {
				    ulmBLAS::mgemm(ni, nj, nb, 1.0,
				                   packA(i, kk), packB(j, kk),
				                   1.0, C, Index(1), nb);
				}
			    }
			    for (Index jj = 0; jj < nj; ++jj) {
				double *a = &Aij[jj * cs];
				const double *c = &C[jj * nb];
				for (Index ii = (i == j ? jj : 0); ii < ni;
				     ++ii) {
				    a[ii * rs] -= c[ii];
				}
			    }
			}
			CHECKDEPS_END(ru);
			taskLeave();
		    }
		}
	    }
	}
	CHECKDEPS_TIME_SET(tGenEnd);
    }
    CHECKDEPS_TIME(tEnd);
    CHECKDEPS_PROFILE(log, threads, tStart, tGenEnd, tEnd);
    if (CHECKDEPS_CHECK(log, "dpotrf outer")) {
	std::abort();
    }
    return info;
}

} // namespace

} // namespace ulmlapack

namespace ulmlapack {

Index
potrf(bool lower, Index n, double *A, Index incRowA, Index incColA,
      const ulmlapack_param *param)
{
    if (n <= 0) {
	return 0;
    }
    ulmlapack_param p;
    if (param) {
	p = *param;
    } else {
	ulmlapack_param_init(&p);
    }
    ulmlapack_dpotrf_resolve(int(n), &p);

    // the upper variant is the lower one of the transposed view
    const Index rs = lower ? incRowA : incColA;
    const Index cs = lower ? incColA : incRowA;

    if (n <= p.serial_n || p.threads == 1 || n <= p.nb) {
	return potrf_lower_packed(n, A, rs, cs);
    }
    typedef ulmBLAS::BlockSize<double> BS;
    if (p.outer > 1 && p.nb <= BS::MC && p.nb <= BS::NC && p.nb <= BS::KC) {
	return potrf_outer_tasks(n, stridedTiles(A, rs, cs, p.nb), p.nb,
	                         p.outer, std::max(1, p.window), p.strip,
	                         p.threads, p.empty_tasks != 0);
    }
    return potrf_tasks(n, stridedTiles(A, rs, cs, p.nb), 0, p.nb, p.threads,
                       p.empty_tasks != 0, p.prepack != 0);
}

namespace {

//  parameters and the transposed view, as in potrf
bool
potrfSetup(bool lower, Index n, Index incRowA, Index incColA,
           const ulmlapack_param *param, ulmlapack_param &p, Index &rs,
           Index &cs)
{
    if (param) {
	p = *param;
    } else {
	ulmlapack_param_init(&p);
    }
    ulmlapack_dpotrf_resolve(int(n), &p);
    rs = lower ? incRowA : incColA;
    cs = lower ? incColA : incRowA;
    return n <= p.serial_n || p.threads == 1 || n <= p.nb; // serial
}

} // namespace

Index
potrf_colmajor(bool lower, Index n, double *A, Index incRowA, Index incColA,
               const ulmlapack_param *param)
{
    if (n <= 0) {
	return 0;
    }
    ulmlapack_param p;
    Index rs, cs;
    if (potrfSetup(lower, n, incRowA, incColA, param, p, rs, cs)) {
	return potrf_lower_packed(n, A, rs, cs);
    }
    double *B = workspace(std::size_t(n) * n);
    const Source src = {A, rs, cs};
    return potrf_tasks(n, stridedTiles(B, 1, n, p.nb), &src, p.nb, p.threads,
                       p.empty_tasks != 0, p.prepack != 0);
}

Index
potrf_tiled(bool lower, Index n, double *A, Index incRowA, Index incColA,
            const ulmlapack_param *param)
{
    if (n <= 0) {
	return 0;
    }
    ulmlapack_param p;
    Index rs, cs;
    if (potrfSetup(lower, n, incRowA, incColA, param, p, rs, cs)) {
	return potrf_lower_packed(n, A, rs, cs);
    }
    const Index nt = (n + p.nb - 1) / p.nb;
    double *B = workspace(std::size_t(nt) * nt * p.nb * p.nb);
    const Source src = {A, rs, cs};
    return potrf_tasks(n, tiledTiles(B, nt, p.nb), &src, p.nb, p.threads,
                       p.empty_tasks != 0, p.prepack != 0);
}

Index
potrf(TileMatrix &A, const ulmlapack_param *param)
{
    const Index n = A.n;
    if (n <= 0) {
	return 0;
    }
    ulmlapack_param p;
    if (param) {
	p = *param;
    } else {
	ulmlapack_param_init(&p);
    }
    p.nb = int(A.nb);
    ulmlapack_dpotrf_resolve(int(n), &p);
    const Tiles W = tiledTiles(A.tile(0, 0), A.mt, A.nb);
    if (A.nt == 1) {
	return potrf_lower_rec(n, W.tile(0, 0), 1, A.nb);
    }
    //  no serial algorithm across tiles: the task graph (also with one
    //  thread)
    return potrf_tasks(n, W, 0, A.nb, p.threads, p.empty_tasks != 0,
                       p.prepack != 0);
}

} // namespace ulmlapack

using ulmlapack::Index;

extern "C" {

void
ulmlapack_param_init(ulmlapack_param *p)
{
    p->nb = 0;
    p->ib = 0;
    p->panel_threads = 0;
    p->threads = 0;
    p->serial_n = -1;
    p->empty_tasks = 0;
    p->prepack = 0;
    p->outer = 0;
    p->window = 1;
    p->strip = 0;
}

void
ulmlapack_dpotrf_resolve(int n, ulmlapack_param *p)
{
    // Defaults tuned on one socket of a Xeon Platinum 8358 (Ice Lake, 32
    // cores) with sizes off the benchmark grid; see README (Tuning).
    if (p->threads <= 0) {
	p->threads = omp_get_max_threads();
    }
    if (p->nb <= 0) {
	p->nb = (n < 250)    ? 32
	        : (n < 600)  ? 48
	        : (n < 900)  ? 64
	        : (n < 2200) ? 96
	                     : 128;
    }
    if (p->serial_n < 0) {
	p->serial_n = 100;
    }
    // about one thread per 64 rows (at least 4): more threads only add
    // idle time and synchronization for small and medium n
    p->threads = std::max(1, std::min(p->threads, std::max(4, n / 64)));
    // more threads than tiles of the trailing matrix do not help
    const Index nt = (n + p->nb - 1) / p->nb;
    const Index tiles = nt * (nt + 1) / 2;
    p->threads = (int)std::max<Index>(1, std::min<Index>(p->threads, tiles));
}

int
ulmlapack_dpotrf_param(char uplo, int n, double *A, int ldA,
                       const ulmlapack_param *param)
{
    const bool lower = (uplo == 'L' || uplo == 'l');
    if (!lower && uplo != 'U' && uplo != 'u') {
	return -1;
    }
    if (n < 0) {
	return -2;
    }
    if (ldA < std::max(1, n)) {
	return -4;
    }
    return (int)ulmlapack::potrf(lower, n, A, Index(1), Index(ldA), param);
}

int
ulmlapack_dpotrf(char uplo, int n, double *A, int ldA)
{
    return ulmlapack_dpotrf_param(uplo, n, A, ldA, 0);
}

} // extern "C"
