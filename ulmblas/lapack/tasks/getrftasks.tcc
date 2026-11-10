#ifndef ULMBLAS_LAPACK_TASKS_GETRFTASKS_TCC
#define ULMBLAS_LAPACK_TASKS_GETRFTASKS_TCC 1

//
//  The matrix is divided into nb x nb tiles; a byte per tile (tok) stands
//  for the tile in the depend clauses.  Step k (panel = column block k,
//  rows k*nb, ..., m-1):
//
//      PANEL(k)       LU with partial pivoting of the panel      inout tiles
//      (i,k), i >= k SWAP_TRSM(k,j) interchanges of panel k in column block j,
//                     U(k,j) = L(k,k)^{-1} A(k,j)                in (k,k),
//                     inout (i,j), i >= k
//      GEMM(k,i,j)    A(i,j) -= L(i,k) U(k,j)                    in (i,k),
//      (k,j), inout (i,j) LSWAP(k,c)     interchanges of panel k in an earlier
//                     column block c ("pivoting to the left")    in (k,k),
//                     inout (i,c), i >= k
//
//  The interchanges of a panel act on all rows below k*nb, so the tasks
//  that apply them depend on the whole column of tiles (OpenMP 5.0
//  iterator in the depend clause); the GEMMs depend on single tiles and run
//  in parallel.  The pivoting to the left runs as soon as a panel is done.
//
//  Priorities: PANEL > tasks of column block k+1 (lookahead) > other
//  updates > LSWAP.
//
//  The panel is the critical path.  It is factored either by the serial
//  recursive algorithm (getrf2) or by a team of threads: every thread of
//  the team owns a contiguous range of rows of the panel and keeps it in
//  its cache; per column the threads search their rows for the pivot
//  candidate, meet at a barrier, and all compute the pivot; the interchange
//  and the update of the own rows follow without another barrier.  The
//  panel is processed in blocks of ib columns.
//
//  The team is formed dynamically: the panel task creates helper tasks and
//  waits at most joinWait seconds for helpers to join; then it closes the
//  team.  Helpers that start later, or that run in the panel's own thread,
//  return at once.  So the panel uses the threads that are idle when it
//  starts and never waits for a thread that is not running.
//

#ifdef _OPENMP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <utility>
#include <omp.h>
#include <ulmblas/auxiliary/abs1.h>
#include <ulmblas/lapack/buffer.h>
#include <ulmblas/lapack/getrf2.h>
#include <ulmblas/lapack/safemin.h>
#include <ulmblas/lapack/tasks/getrftasks.h>
#include <ulmblas/lapack/tasks/sync.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trlsm.h>

namespace ulmBLAS {
namespace tasks {

namespace getrf_impl {

const int PRIO_PANEL = 3;
const int PRIO_NEXT = 2;
const int PRIO_UPD = 1;
const int PRIO_LSWAP = 0;

const int CLOSED = 1 << 30;

template <typename IndexType, typename T> struct Panel
{
    typedef decltype(std::real(T())) Real;

    // panel
    IndexType m, n;
    T *A;
    IndexType rs, cs;   // row and column stride
    T *orig;            // copy: the panel itself (else null),
    IndexType ors, ocs; // A is a column-major copy (rs 1, cs m)
    IndexType *piv;
    IndexType ib;

    // team
    int creator;  // omp thread number of the panel task
    int excluded; // thread that must not join (or -1)
    int maxSize;
    std::atomic<int> state;   // members (bits 0..29), closed (bit 30)
    std::atomic<int> started; // set when the team is closed and set up
    int size;                 // final size
    IndexType chunk;          // rows per member
    SpinBarrier barrier;

    // per column, double buffered: candidates of the members
    Buffer<Real> candVal;      // [2][size]
    Buffer<IndexType> candIdx; // [2][size]
    Buffer<T> candRow;         // [2][size][ib]
    Buffer<T> topRow;          // [2][ib]

    IndexType info; // written by member 0
};

//  Try to join; returns the member number or -1.
template <typename IndexType, typename T>
int
join(Panel<IndexType, T> &P)
{
    int s = P.state.load(std::memory_order_acquire);
    while (!(s & CLOSED) && s < P.maxSize) {
	if (P.state.compare_exchange_weak(s, s + 1,
	                                  std::memory_order_acq_rel)) {
	    return s; // members 1, 2, ... (0 is the creator)
	}
    }
    return -1;
}

//
//  One member of a panel team.  UnitRows: row stride 1, so that the loops
//  over the rows of a column are contiguous.
//
template <bool UnitRows, typename IndexType, typename T>
void
member(Panel<IndexType, T> &P, int r)
{
    typedef typename Panel<IndexType, T>::Real Real;

    const IndexType m = P.m;
    const IndexType n = P.n;
    const IndexType kmin = std::min(m, n);
    const IndexType ib = P.ib;
    const IndexType rs = UnitRows ? IndexType(1) : P.rs;
    const IndexType cs = P.cs;
    const int size = P.size;
    const IndexType lo = std::min(m, r * P.chunk);
    const IndexType hi = std::min(m, lo + P.chunk);
    const Real sfmin = safeMin<Real>();
    T *A = P.A;
    int sense = 0;

    if (P.orig) {
	for (IndexType j = 0; j < n; ++j) {
	    for (IndexType i = lo; i < hi; ++i) {
		A[i * rs + j * cs] = P.orig[i * P.ors + j * P.ocs];
	    }
	}
    }

    for (IndexType c = 0; c < kmin; c += ib) {
	const IndexType kb = std::min(ib, kmin - c);

	for (IndexType j = c; j < c + kb; ++j) {
	    const int par = int(j & 1);

	    // pivot candidate among the own rows (first maximum)
	    const IndexType i0 = std::max(lo, j);
	    IndexType best = -1;
	    Real bval = 0;
	    if (i0 < hi) {
		const T *a = &A[j * cs];
		best = i0;
		bval = abs1(a[i0 * rs]);
		for (IndexType i = i0 + 1; i < hi; ++i) {
		    const Real v = abs1(a[i * rs]);
		    if (v > bval) {
			bval = v;
			best = i;
		    }
		}
		T *cr = &P.candRow[(par * size + r) * ib];
		for (IndexType l = 0; l < kb; ++l) {
		    cr[l] = A[best * rs + (c + l) * cs];
		}
	    }
	    P.candIdx[par * size + r] = best;
	    P.candVal[par * size + r] = bval;
	    if (lo <= j && j < hi) {
		T *tr = &P.topRow[par * ib];
		for (IndexType l = 0; l < kb; ++l) {
		    tr[l] = A[j * rs + (c + l) * cs];
		}
	    }

	    P.barrier.wait(sense);

	    // pivot: the first maximum over the members in row order
	    int ws = -1;
	    Real pv = -1;
	    for (int s = 0; s < size; ++s) {
		if (P.candIdx[par * size + s] >= 0 &&
		    P.candVal[par * size + s] > pv) {
		    pv = P.candVal[par * size + s];
		    ws = s;
		}
	    }
	    const IndexType p = P.candIdx[par * size + ws];
	    const T *prow = &P.candRow[(par * size + ws) * ib];
	    const T pivot = prow[j - c];

	    if (r == 0) {
		P.piv[j] = p;
	    }
	    if (pivot != T(0)) {
		if (p != j) {
		    if (lo <= j && j < hi) {
			for (IndexType l = 0; l < kb; ++l) {
			    A[j * rs + (c + l) * cs] = prow[l];
			}
		    }
		    if (lo <= p && p < hi) {
			const T *tr = &P.topRow[par * ib];
			for (IndexType l = 0; l < kb; ++l) {
			    A[p * rs + (c + l) * cs] = tr[l];
			}
		    }
		}
		const IndexType s0 = std::max(lo, j + 1);
		if (s0 < hi) {
		    T *a = &A[j * cs];
		    if (std::abs(pivot) >= sfmin) {
			const T rcp = T(1) / pivot;
			for (IndexType i = s0; i < hi; ++i) {
			    a[i * rs] *= rcp;
			}
		    } else {
			for (IndexType i = s0; i < hi; ++i) {
			    a[i * rs] /= pivot;
			}
		    }
		    for (IndexType l = j + 1; l < c + kb; ++l) {
			const T u = prow[l - c];
			if (u != T(0)) {
			    T *b = &A[l * cs];
			    for (IndexType i = s0; i < hi; ++i) {
				b[i * rs] -= a[i * rs] * u;
			    }
			}
		    }
		}
	    } else if (r == 0 && P.info == 0) {
		P.info = j + 1;
	    }
	}

	P.barrier.wait(sense);

	//  Interchanges of this block in the columns outside of it, and the
	//  triangular solve for the columns right of it; columns split
	//  among the members.
	const IndexType nOut = c + (n - c - kb);
	const IndexType per = (nOut + size - 1) / size;
	const IndexType q0 = std::min(nOut, r * per);
	const IndexType q1 = std::min(nOut, q0 + per);
	for (IndexType q = q0; q < q1; ++q) {
	    const IndexType col = (q < c) ? q : q + kb;
	    T *a = &A[col * cs];
	    for (IndexType j = c; j < c + kb; ++j) {
		const IndexType p = P.piv[j];
		if (p != j) {
		    std::swap(a[j * rs], a[p * rs]);
		}
	    }
	    if (col >= c + kb) {
		for (IndexType j = c; j < c + kb; ++j) {
		    const T x = a[j * rs];
		    if (x != T(0)) {
			const T *l = &A[j * cs];
			for (IndexType i = j + 1; i < c + kb; ++i) {
			    a[i * rs] -= l[i * rs] * x;
			}
		    }
		}
	    }
	}

	P.barrier.wait(sense);

	// update of the own rows right of the block
	const IndexType g0 = std::max(lo, c + kb);
	if (g0 < hi && c + kb < n) {
	    gemm(hi - g0, n - c - kb, kb, T(-1), false, &A[g0 * rs + c * cs],
	         rs, cs, false, &A[c * rs + (c + kb) * cs], rs, cs, T(1),
	         &A[g0 * rs + (c + kb) * cs], rs, cs);
	}
    }

    if (P.orig) {
	// the swaps of the last block wrote rows of other members
	P.barrier.wait(sense);
	for (IndexType j = 0; j < n; ++j) {
	    for (IndexType i = lo; i < hi; ++i) {
		P.orig[i * P.ors + j * P.ocs] = A[i * rs + j * cs];
	    }
	}
    }
}

template <typename IndexType, typename T>
void
member(Panel<IndexType, T> &P, int r)
{
    if (P.rs == 1) {
	member<true>(P, r);
    } else {
	member<false>(P, r);
    }
}

//
//  Factor the panel (m x n).  piv: 0-based, relative to the panel.
//  Returns 0 or the first zero pivot (1-based).
//
template <typename IndexType, typename T>
IndexType
panel(IndexType m, IndexType n, T *A, IndexType rs, IndexType cs,
      IndexType *piv, IndexType ib, int maxTeam, IndexType minRows,
      double joinWait, int excluded)
{
    int maxSize =
        int(std::min<IndexType>(maxTeam, std::max<IndexType>(1, m / minRows)));
    if (maxSize <= 1 || n <= 1) {
	return getrf2(m, n, A, rs, cs, piv, IndexType(1));
    }

    typedef Panel<IndexType, T> P_;
    P_ P;
    P.m = m;
    P.n = n;
    P.A = A;
    P.rs = rs;
    P.cs = cs;
    P.orig = 0;
    P.ors = 0;
    P.ocs = 0;
    P.piv = piv;
    P.ib = std::min(ib, std::min(m, n));
    P.creator = omp_get_thread_num();
    P.excluded = excluded;
    P.maxSize = maxSize;
    P.state.store(1);
    P.started.store(0);
    P.size = 1;
    P.info = 0;

    P_ *pp = &P;
    for (int h = 1; h < maxSize; ++h) {
#pragma omp task firstprivate(pp) priority(PRIO_PANEL)
	{
	    const int t = omp_get_thread_num();
	    if (t != pp->creator && t != pp->excluded) {
		const int r = join(*pp);
		if (r > 0) {
		    // wait until the team is closed and set up
		    while (pp->started.load(std::memory_order_acquire) == 0) {
			cpuRelax();
		    }
		    member(*pp, r);
		}
	    }
	}
    }

    // wait for helpers (at most joinWait seconds), then close the team
    const double t0 = seconds();
    while ((P.state.load(std::memory_order_acquire) & ~CLOSED) < maxSize &&
           seconds() - t0 < joinWait) {
	cpuRelax();
    }
    const int size =
        P.state.fetch_or(CLOSED, std::memory_order_acq_rel) & ~CLOSED;

    //  copy of the panel for a team; it must outlive the members, which
    //  finish before the taskwait
    Buffer<T> copy;
    IndexType info;
    if (size == 1) {
	P.started.store(1, std::memory_order_release);
	info = getrf2(m, n, A, rs, cs, piv, IndexType(1));
    } else {
	IndexType chunk = (m + size - 1) / size;
	chunk = (chunk + 7) / 8 * 8;
	P.size = size;
	P.chunk = chunk;
	//  neither stride 1: the team works on a column-major copy of the
	//  panel; every member copies its own rows in and back
	if (rs != 1 && cs != 1) {
	    copy.resize(std::size_t(m) * std::size_t(n));
	    P.orig = A;
	    P.ors = rs;
	    P.ocs = cs;
	    P.A = copy.data();
	    P.rs = 1;
	    P.cs = m;
	}
	P.barrier.reset(size);
	P.candVal.resize(2 * size);
	P.candVal.fill(typename P_::Real(0));
	P.candIdx.resize(2 * size);
	P.candIdx.fill(IndexType(-1));
	P.candRow.resize(2 * size * P.ib);
	P.candRow.fill(T(0));
	P.topRow.resize(2 * P.ib);
	P.topRow.fill(T(0));
	P.started.store(1, std::memory_order_release);

	member(P, 0);
	info = P.info;
    }
#pragma omp taskwait
    return info;
}

} // namespace getrf_impl

template <typename IndexType, typename T>
IndexType
getrfTasks(IndexType m, IndexType n, T *A, IndexType incRowA, IndexType incColA,
           IndexType *piv, IndexType nb, IndexType ib, int panelThreads,
           int threads)
{
    using namespace getrf_impl;

    const IndexType rs = incRowA, cs = incColA;
    const IndexType mt = (m + nb - 1) / nb;
    const IndexType nt = (n + nb - 1) / nb;
    const IndexType kmin = std::min(m, n);
    const IndexType kt = (kmin + nb - 1) / nb;

    //  panel teams: at least minRows rows per member; wait at most joinWait
    //  seconds for members
    const IndexType minRows = 64;
    const double joinWait = 20e-6;

    Buffer<char> tok(mt * nt);
    Buffer<IndexType> pinfo(kt);
    pinfo.fill(IndexType(0));
    char *Tk = tok.data();
    (void)Tk; // only used in depend clauses (GCC warns)
    IndexType *PI = pinfo.data();

    // the generating thread does not join panel teams while it creates tasks
    std::atomic<int> generator(-1);

#pragma omp parallel num_threads(threads) shared(generator)
#pragma omp single
    {
	generator.store(omp_get_thread_num());

	for (IndexType k = 0; k < kt; ++k) {
	    const IndexType r0 = k * nb;
	    const IndexType mp = m - r0;
	    const IndexType wp = std::min(nb, n - r0);
	    const IndexType kp = std::min(mp, wp);
	    T *Akk = &A[r0 * (rs + cs)];
	    IndexType *pk = &piv[r0];

#pragma omp task depend(iterator(ii = k : mt), inout : Tk[ii * nt + k])        \
    priority(PRIO_PANEL)
	    {
		taskEnter();
		PI[k] = panel(mp, wp, Akk, rs, cs, pk, ib, panelThreads,
		              minRows, joinWait, generator.load());
		taskLeave();
	    }

	    // pivoting to the left
	    for (IndexType c = 0; c < k; ++c) {
		T *Ac = &A[r0 * rs + c * nb * cs];

#pragma omp task depend(in : Tk[k * nt + k])                                   \
    depend(iterator(ii = k : mt), inout : Tk[ii * nt + c])                     \
    priority(PRIO_LSWAP)
		{
		    taskEnter();
		    getrf2_impl::swapRows(nb, Ac, rs, cs, IndexType(0), kp, pk);
		    taskLeave();
		}
	    }

	    // trailing matrix
	    for (IndexType j = k + 1; j < nt; ++j) {
		const IndexType wj = std::min(nb, n - j * nb);
		const int pri = (j == k + 1) ? PRIO_NEXT : PRIO_UPD;
		T *Akj = &A[r0 * rs + j * nb * cs];

#pragma omp task depend(in : Tk[k * nt + k])                                   \
    depend(iterator(ii = k : mt), inout : Tk[ii * nt + j]) priority(pri)
		{
		    taskEnter();
		    getrf2_impl::swapRows(wj, Akj, rs, cs, IndexType(0), kp,
		                          pk);
		    trlsm(kp, wj, T(1), false, true, Akk, rs, cs, Akj, rs, cs);
		    taskLeave();
		}
		for (IndexType i = k + 1; i < mt; ++i) {
		    const IndexType mi = std::min(nb, m - i * nb);
		    const T *Aik = &A[i * nb * rs + r0 * cs];
		    T *Aij = &A[i * nb * rs + j * nb * cs];

#pragma omp task depend(in : Tk[i * nt + k], Tk[k * nt + j])                   \
    depend(inout : Tk[i * nt + j]) priority(pri)
		    {
			taskEnter();
			gemm(mi, wj, kp, T(-1), false, Aik, rs, cs, false, Akj,
			     rs, cs, T(1), Aij, rs, cs);
			taskLeave();
		    }
		}
	    }
	}
	generator.store(-1);
    }

    IndexType info = 0;
    for (IndexType k = 0; k < kt; ++k) {
	if (PI[k]) {
	    info = k * nb + PI[k];
	    break;
	}
    }
    for (IndexType k = 0; k < kt; ++k) {
	const IndexType r0 = k * nb;
	const IndexType kp = std::min(m - r0, std::min(nb, n - r0));
	for (IndexType l = 0; l < kp; ++l) {
	    piv[r0 + l] += r0;
	}
    }
    return info;
}

} // namespace tasks
} // namespace ulmBLAS

#endif // _OPENMP

#endif // ULMBLAS_LAPACK_TASKS_GETRFTASKS_TCC
