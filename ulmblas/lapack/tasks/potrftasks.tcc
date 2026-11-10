#ifndef ULMBLAS_LAPACK_TASKS_POTRFTASKS_TCC
#define ULMBLAS_LAPACK_TASKS_POTRFTASKS_TCC 1

//
//  The tiles are nb x nb blocks of A; a byte per tile (tok) stands for the
//  tile in the depend clauses:
//
//      for k = 0, ..., nt-1
//          POTRF  A(k,k) = L(k,k) L(k,k)^H             inout A(k,k)
//          for i > k
//              TRSM   A(i,k) = A(i,k) L(k,k)^{-H}      in A(k,k), inout A(i,k)
//          for i > k
//              HERK   A(i,i) -= A(i,k) A(i,k)^H        in A(i,k), inout A(i,i)
//              for k < j < i
//                  GEMM  A(i,j) -= A(i,k) A(j,k)^H     in A(i,k), A(j,k),
//                                                      inout A(i,j)
//
//  The tasks on the critical path (POTRF, the TRSMs of column k and the
//  updates of column k+1) have higher priorities: the factorization of the
//  next column can start while the rest of the trailing matrix is updated
//  (lookahead by priority, as in PLASMA).  Priorities only take effect if
//  OMP_MAX_TASK_PRIORITY is set (e.g. to 3).
//

#include <algorithm>
#include <atomic>
#include <limits>
#include <ulmblas/lapack/buffer.h>
#include <ulmblas/lapack/potrf2.h>
#include <ulmblas/lapack/tasks/potrftasks.h>
#include <ulmblas/lapack/tasks/sync.h>
#include <ulmblas/level3/gemm.h>

namespace ulmBLAS {
namespace tasks {

template <typename IndexType, typename T>
IndexType
potrfTasks(IndexType n, T *A, IndexType incRowA, IndexType incColA,
           IndexType nb, int threads)
{
    const int PRIO_POTRF = 3;
    const int PRIO_CRIT = 2;
    const int PRIO_NEXT = 1;
    const int PRIO_OTHER = 0;

    const IndexType nt = (n + nb - 1) / nb;
    const IndexType rs = incRowA, cs = incColA;
    auto tile = [=](IndexType i, IndexType j) {
	return &A[i * nb * rs + j * nb * cs];
    };

    Buffer<char> tok(nt * nt);
    char *Tk = tok.data();
    (void)Tk;      // only used in depend clauses (GCC warns)
    (void)threads; // without OpenMP

    //  step of the first failed POTRF (max: none) and its INFO; tasks of
    //  later steps are skipped
    std::atomic<IndexType> failStep(std::numeric_limits<IndexType>::max());
    IndexType info = 0;
    auto run = [&](IndexType k) {
	return failStep.load(std::memory_order_relaxed) > k;
    };
    (void)PRIO_POTRF;
    (void)PRIO_CRIT;
    (void)PRIO_NEXT;
    (void)PRIO_OTHER;

#pragma omp parallel num_threads(threads) shared(failStep, info)
#pragma omp single
    {
	for (IndexType k = 0; k < nt; ++k) {
	    const IndexType nk = std::min(nb, n - k * nb);
	    T *Akk = tile(k, k);

#pragma omp task depend(inout : Tk[k * nt + k]) priority(PRIO_POTRF)
	    {
		taskEnter();
		if (run(k)) {
		    const IndexType inf = potrf2(nk, Akk, rs, cs);
		    if (inf) {
			//  tasks of later steps depend on this one, so the
			//  first failure is the one with the smallest k
			info = k * nb + inf;
			failStep.store(k);
		    }
		}
		taskLeave();
	    }
	    for (IndexType i = k + 1; i < nt; ++i) {
		const IndexType ni = std::min(nb, n - i * nb);
		T *Aik = tile(i, k);

#pragma omp task depend(in : Tk[k * nt + k]) depend(inout : Tk[i * nt + k])    \
    priority(i == k + 1 ? PRIO_POTRF : PRIO_CRIT)
		{
		    taskEnter();
		    if (run(k)) {
			potrf2_impl::trsmRightLowerConjTrans(nk, ni, Akk, rs,
			                                     cs, Aik, rs, cs);
		    }
		    taskLeave();
		}
	    }
	    for (IndexType i = k + 1; i < nt; ++i) {
		const IndexType ni = std::min(nb, n - i * nb);
		const T *Aik = tile(i, k);
		T *Aii = tile(i, i);

#pragma omp task depend(in : Tk[i * nt + k]) depend(inout : Tk[i * nt + i])    \
    priority(i == k + 1 ? PRIO_POTRF : PRIO_NEXT)
		{
		    taskEnter();
		    if (run(k)) {
			potrf2_impl::herkLower(ni, nk, Aik, rs, cs, Aii, rs,
			                       cs);
		    }
		    taskLeave();
		}
		for (IndexType j = k + 1; j < i; ++j) {
		    const IndexType nj = std::min(nb, n - j * nb);
		    const T *Ajk = tile(j, k);
		    T *Aij = tile(i, j);

#pragma omp task depend(in : Tk[i * nt + k], Tk[j * nt + k])                   \
    depend(inout : Tk[i * nt + j])                                             \
    priority(j == k + 1 ? PRIO_CRIT : PRIO_OTHER)
		    {
			taskEnter();
			if (run(k)) {
			    gemm(ni, nj, nk, T(-1), false, Aik, rs, cs, true,
			         Ajk, cs, rs, T(1), Aij, rs, cs);
			}
			taskLeave();
		    }
		}
	    }
	}
    }
    return info;
}

} // namespace tasks
} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_POTRFTASKS_TCC
