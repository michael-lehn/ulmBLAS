#ifndef ULMBLAS_LAPACK_GETRFLA_TCC
#define ULMBLAS_LAPACK_GETRFLA_TCC 1

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/lapack/getrf2.h>
#include <ulmblas/lapack/getrfla.h>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trlsm.h>

namespace ulmBLAS {

namespace getrfla_impl {

//
//  Panel widths: nb, or adaptive (nbMin > 0): a share 1 - sqrt(1 - 1/nt)
//  of the remaining columns (as OpenBLAS's FORMULA2: the update of the next
//  panel and its factorization on one thread take about as long as the
//  trailing update on the others), between nbMin and nb, multiples of 8.
//  The widths depend only on the position, every thread computes them.
//
template <typename IndexType, typename T, typename PivIndex> struct Steps
{
    IndexType m, n, nb, nbMin, chunk;
    double share; // 1 - sqrt(1 - 1/nt)
    T *A;
    IndexType rs, cs;
    PivIndex *piv;
    IndexType incPiv;

    std::atomic<std::ptrdiff_t> arrived;
    //  per step (parity of K): tasks taken of the next block column and
    //  columns taken of the trailing matrix, tasks done of the next block
    //  column
    std::atomic<IndexType> rowTask[2], trailTask[2], rowsDone[2];
    //  K+1 once U(K, K+1) is solved
    std::atomic<IndexType> uReady;
    std::atomic<IndexType> leftTask;
    std::atomic<IndexType> info;

    IndexType
    kmin() const
    {
	return std::min(m, n);
    }

    //  width of the panel starting at s (s < kmin)
    IndexType
    width(IndexType s) const
    {
	IndexType w = nb;
	if (nbMin > 0) {
	    w = IndexType(share * double(n - s)) + 1;
	    w = std::max(nbMin, std::min(nb, (w + 7) / 8 * 8));
	}
	return std::min(w, kmin() - s);
    }

    //  all nt threads at sync point s (counter, never reset)
    void
    sync(std::ptrdiff_t s, std::ptrdiff_t nt)
    {
	arrived.fetch_add(1, std::memory_order_acq_rel);
	threads::Backoff backoff;
	while (arrived.load(std::memory_order_acquire) < (s + 1) * nt) {
	    backoff.pause();
	}
    }

    //  columns [c0, c1) with the panel at r0 (kb columns): interchanges and
    //  U = L^{-1} A(r0:r0+kb, c)
    void
    solve(IndexType r0, IndexType kb, IndexType c0, IndexType c1)
    {
	T *B = &A[c0 * cs];

	getrf2_impl::swapRows(c1 - c0, B, rs, cs, r0, r0 + kb, piv, incPiv);
	trlsm(kb, c1 - c0, T(1), false, true, &A[r0 * (rs + cs)], rs, cs,
	      &B[r0 * rs], rs, cs);
    }

    //  rows [i0, i1) (below the panel) of columns [c0, c1): A -= L U
    void
    rank(IndexType r0, IndexType kb, IndexType i0, IndexType i1, IndexType c0,
         IndexType c1)
    {
	if (i1 > i0 && c1 > c0) {
	    gemm(i1 - i0, c1 - c0, kb, T(-1), false, &A[i0 * rs + r0 * cs], rs,
	         cs, false, &A[r0 * rs + c0 * cs], rs, cs, T(1),
	         &A[i0 * rs + c0 * cs], rs, cs);
	}
    }

    //  the panel at r0 (kb columns, rows r0, ..., m-1) factored; pivots made
    //  absolute
    void
    factor(IndexType r0, IndexType kb)
    {
	PivIndex *p = &piv[r0 * incPiv];

	const IndexType i =
	    getrf2(m - r0, kb, &A[r0 * (rs + cs)], rs, cs, p, incPiv);
	for (IndexType l = 0; l < kb; ++l) {
	    p[l * incPiv] += PivIndex(r0);
	}
	if (i && info.load(std::memory_order_relaxed) == 0) {
	    info.store(r0 + i, std::memory_order_relaxed);
	}
    }

    //  trailing columns [c0, c0 + ncols) with the panel at r0: guided
    //  chunks of columns (a share of the remaining columns, at least chunk,
    //  multiples of 8), each solved and updated on one thread
    void
    trailingGuided(IndexType r0, IndexType kb, IndexType c0, IndexType ncols,
                   int p, std::ptrdiff_t nt)
    {
	const IndexType below = r0 + kb;
	const IndexType minW = chunk > 0 ? chunk : IndexType(64);
	std::atomic<IndexType> &taken = trailTask[p];
	IndexType c = taken.load(std::memory_order_relaxed);
	while (c < ncols) {
	    IndexType w = (ncols - c) / IndexType(nt);
	    w = (std::max(w, minW) + 7) / 8 * 8;
	    w = std::min(w, ncols - c);
	    if (taken.compare_exchange_weak(c, c + w,
	                                    std::memory_order_relaxed)) {
		solve(r0, kb, c0 + c, c0 + c + w);
		rank(r0, kb, below, m, c0 + c, c0 + c + w);
		c = taken.load(std::memory_order_relaxed);
	    }
	}
    }

    //  step K: panel K at r0 (kb columns, factored), the next panel at r1
    //  (kb1 columns, kb1 = 0: none)
    void
    step(IndexType K, IndexType r0, IndexType kb, IndexType r1, IndexType kb1,
         std::ptrdiff_t tid, std::ptrdiff_t nt)
    {
	const int p = int(K % 2);
	const IndexType below = r0 + kb;

	if (kb1 > 0) {
	    //  the next block column: U on thread 0, the rows below on all
	    //  threads, then the panel on thread 0
	    if (tid == 0) {
		solve(r0, kb, r1, r1 + kb1);
		uReady.store(K + 1, std::memory_order_release);
	    }
	    const IndexType mr = m - below;
	    IndexType h = (mr + IndexType(nt) - 1) / IndexType(nt);
	    h = std::max(IndexType(64), (h + 15) / 16 * 16);
	    const IndexType rowTasks = (mr + h - 1) / h;
	    for (IndexType t =
	             rowTask[p].fetch_add(1, std::memory_order_relaxed);
	         t < rowTasks;
	         t = rowTask[p].fetch_add(1, std::memory_order_relaxed)) {
		threads::Backoff backoff;
		while (uReady.load(std::memory_order_acquire) < K + 1) {
		    backoff.pause();
		}
		const IndexType i0 = below + t * h;
		rank(r0, kb, i0, std::min(m, i0 + h), r1, r1 + kb1);
		rowsDone[p].fetch_add(1, std::memory_order_acq_rel);
	    }
	    if (tid == 0) {
		threads::Backoff backoff;
		while (rowsDone[p].load(std::memory_order_acquire) < rowTasks) {
		    backoff.pause();
		}
		factor(r1, kb1);
	    }
	}
	const IndexType c0 = kb1 > 0 ? r1 + kb1 : kmin();
	const IndexType ncols = n - c0;
	trailingGuided(r0, kb, c0, ncols, p, nt);
    }

    void
    operator()(std::ptrdiff_t tid, std::ptrdiff_t nt)
    {
	const IndexType k = kmin();
	std::ptrdiff_t s = 0;

	IndexType r0 = 0, kb = width(0);
	if (tid == 0) {
	    factor(r0, kb);
	}
	sync(s++, nt);
	for (IndexType K = 0; r0 < k; ++K, ++s) {
	    const IndexType r1 = r0 + kb;
	    const IndexType kb1 = r1 < k ? width(r1) : IndexType(0);
	    step(K, r0, kb, r1, kb1, tid, nt);
	    sync(s, nt);
	    //  the counters of this step are used again two steps later,
	    //  after the next sync
	    if (tid == 0) {
		const int p = int(K % 2);
		rowTask[p].store(0, std::memory_order_relaxed);
		trailTask[p].store(0, std::memory_order_relaxed);
		rowsDone[p].store(0, std::memory_order_relaxed);
	    }
	    r0 = r1;
	    kb = kb1;
	}
	//  interchanges left of the panels (all but the last), panel by panel
	//  (task t: panel t; the numbers taken by a thread increase)
	IndexType walked = 0;
	r0 = 0;
	for (IndexType t = leftTask.fetch_add(1, std::memory_order_relaxed);;
	     t = leftTask.fetch_add(1, std::memory_order_relaxed)) {
	    while (walked < t && r0 < k) {
		r0 += width(r0);
		++walked;
	    }
	    if (r0 >= k) {
		break;
	    }
	    const IndexType w = width(r0);
	    if (r0 + w >= k) {
		break;
	    }
	    getrf2_impl::swapRows(w, &A[r0 * cs], rs, cs, r0 + w, k, piv,
	                          incPiv);
	}
    }
};

} // namespace getrfla_impl

template <typename IndexType, typename T, typename PivIndex>
IndexType
getrfLookahead(IndexType m, IndexType n, T *A, IndexType incRowA,
               IndexType incColA, PivIndex *piv, IndexType incPiv, IndexType nb,
               IndexType nbMin, IndexType chunk, std::ptrdiff_t nt)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (std::min(m, n) <= 0) {
	return 0;
    }
    if (nb <= 0 || nt <= 1 || std::min(m, n) <= nb) {
	return getrf2(m, n, A, incRowA, incColA, piv, incPiv);
    }
    getrfla_impl::Steps<IndexType, T, PivIndex> steps;
    steps.m = m;
    steps.n = n;
    steps.nb = nb;
    steps.nbMin = nbMin;
    steps.share = 1 - std::sqrt(1 - 1 / double(nt));
    steps.chunk = chunk;
    steps.A = A;
    steps.rs = incRowA;
    steps.cs = incColA;
    steps.piv = piv;
    steps.incPiv = incPiv;
    steps.arrived.store(0);
    for (int p = 0; p < 2; ++p) {
	steps.rowTask[p].store(0);
	steps.trailTask[p].store(0);
	steps.rowsDone[p].store(0);
    }
    steps.uReady.store(0);
    steps.leftTask.store(0);
    steps.info.store(0);
    ThreadPool::instance().run(nt, steps);
    return steps.info.load();
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRFLA_TCC
