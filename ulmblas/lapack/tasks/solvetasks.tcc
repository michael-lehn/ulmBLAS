#ifndef ULMBLAS_LAPACK_TASKS_SOLVETASKS_TCC
#define ULMBLAS_LAPACK_TASKS_SOLVETASKS_TCC 1

#include <algorithm>
#include <utility>
#include <ulmblas/lapack/buffer.h>
#include <ulmblas/lapack/threads.h>
#include <ulmblas/lapack/tasks/param.h>
#include <ulmblas/lapack/tasks/solvetasks.h>
#include <ulmblas/lapack/tasks/sync.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trlsm.h>
#include <ulmblas/level3/trusm.h>

namespace ulmBLAS {
namespace tasks {

namespace solve_impl {

//  rows i and piv[i] of the m x n matrix B, i = 0..k-1 forward or k-1..0
template <typename IndexType, typename T>
void
swapRows(bool forward, IndexType k, const IndexType *piv, IndexType n, T *B,
         IndexType rs, IndexType cs)
{
    for (IndexType j = 0; j < n; ++j) {
	T *b = &B[j * cs];
	if (forward) {
	    for (IndexType i = 0; i < k; ++i) {
		if (piv[i] != i) {
		    std::swap(b[i * rs], b[piv[i] * rs]);
		}
	    }
	} else {
	    for (IndexType i = k - 1; i >= 0; --i) {
		if (piv[i] != i) {
		    std::swap(b[i * rs], b[piv[i] * rs]);
		}
	    }
	}
    }
}

//  B := op(A)^{-1} B, A m x m triangular, B m x n
template <typename IndexType, typename T>
void
trsm(bool upper, bool conj, bool unitDiag, IndexType m, IndexType n, const T *A,
     IndexType ars, IndexType acs, T *B, IndexType brs, IndexType bcs)
{
    if (upper) {
	trusm(m, n, T(1), conj, unitDiag, A, ars, acs, B, brs, bcs);
    } else {
	trlsm(m, n, T(1), conj, unitDiag, A, ars, acs, B, brs, bcs);
    }
}

} // namespace solve_impl

template <typename IndexType, typename T>
void
solveSteps(IndexType n, IndexType nrhs, const SolveStep<IndexType, T> *steps,
           int numSteps, T *B, IndexType incRowB, IndexType incColB,
           IndexType nb, int threads)
{
    typedef SolveStep<IndexType, T> Step;

    if (n <= 0 || nrhs <= 0) {
	return;
    }
    const IndexType rs = incRowB, cs = incColB;

#ifdef _OPENMP
    if (threads > 1) {
	const int PRIO_TRSM = 2;
	const int PRIO_NEXT = 1;
	const int PRIO_GEMM = 0;

	const IndexType mt = (n + nb - 1) / nb;
	const IndexType nt = (nrhs + nb - 1) / nb;
	Buffer<char> tok(std::size_t(mt * nt));
	char *Tk = tok.data();
	(void)Tk; // only used in depend clauses (GCC warns)

#pragma omp parallel num_threads(threads)
#pragma omp single
	{
	    for (int s = 0; s < numSteps; ++s) {
		const Step st = steps[s];
		for (IndexType j = 0; j < nt; ++j) {
		    const IndexType nj = std::min(nb, nrhs - j * nb);
		    T *Bj = &B[j * nb * cs];

		    if (st.kind != Step::Solve) {
			const bool fwd = st.kind == Step::SwapForward;
#pragma omp task depend(iterator(ii = 0 : mt), inout : Tk[ii * nt + j])        \
    priority(PRIO_TRSM)
			{
			    taskEnter();
			    solve_impl::swapRows(fwd, n, st.piv, nj, Bj, rs,
			                         cs);
			    taskLeave();
			}
			continue;
		    }
		    const IndexType ars = st.incRowA, acs = st.incColA;
		    for (IndexType kk = 0; kk < mt; ++kk) {
			//  lower: tile rows 0, 1, ...; upper: mt-1, ...
			const IndexType k = st.upper ? mt - 1 - kk : kk;
			const IndexType mk = std::min(nb, n - k * nb);
			const T *Akk = &st.A[k * nb * (ars + acs)];
			T *Bkj = &Bj[k * nb * rs];

#pragma omp task depend(inout : Tk[k * nt + j]) priority(PRIO_TRSM)
			{
			    taskEnter();
			    solve_impl::trsm(st.upper, st.conj, st.unitDiag, mk,
			                     nj, Akk, ars, acs, Bkj, rs, cs);
			    taskLeave();
			}
			const IndexType i0 = st.upper ? 0 : k + 1;
			const IndexType i1 = st.upper ? k : mt;
			for (IndexType i = i0; i < i1; ++i) {
			    const IndexType mi = std::min(nb, n - i * nb);
			    const T *Aik = &st.A[i * nb * ars + k * nb * acs];
			    T *Bij = &Bj[i * nb * rs];
			    const bool next =
			        st.upper ? i == k - 1 : i == k + 1;

#pragma omp task depend(in : Tk[k * nt + j]) depend(inout : Tk[i * nt + j])    \
    priority(next ? PRIO_NEXT : PRIO_GEMM)
			    {
				taskEnter();
				gemm(mi, nj, mk, T(-1), st.conj, Aik, ars, acs,
				     false, Bkj, rs, cs, T(1), Bij, rs, cs);
				taskLeave();
			    }
			}
		    }
		}
	    }
	}
	return;
    }
#endif
    (void)nb;
    (void)threads;
    for (int s = 0; s < numSteps; ++s) {
	const Step &st = steps[s];
	if (st.kind == Step::Solve) {
	    solve_impl::trsm(st.upper, st.conj, st.unitDiag, n, nrhs, st.A,
	                     st.incRowA, st.incColA, B, rs, cs);
	} else {
	    solve_impl::swapRows(st.kind == Step::SwapForward, n, st.piv, nrhs,
	                         B, rs, cs);
	}
    }
}

template <typename IndexType, typename T>
void
solve(IndexType n, IndexType nrhs, const SolveStep<IndexType, T> *steps,
      int numSteps, T *B, IndexType incRowB, IndexType incColB)
{
    int threads = 1;
    IndexType nb = n;
#ifdef _OPENMP
    if (!omp_in_parallel()) {
	const Param p = solveParam(long(n), long(nrhs), numThreads());
	if (n > IndexType(p.serialN)) {
	    threads = p.threads;
	    nb = IndexType(p.nb);
	}
    }
#endif
    solveSteps(n, nrhs, steps, numSteps, B, incRowB, incColB, nb, threads);
}

} // namespace tasks
} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_SOLVETASKS_TCC
