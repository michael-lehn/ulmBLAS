#ifndef ULMBLAS_LAPACK_TASKS_INVERSETASKS_TCC
#define ULMBLAS_LAPACK_TASKS_INVERSETASKS_TCC 1

//
//  pztrtri, lower (step k, tiles of A, X = inv(A)):
//
//      TRSM   A(m,k) := -A(m,k) A(k,k)^{-1}    m > k     in (k,k), inout (m,k)
//      GEMM   A(m,n) += A(m,k) A(k,n)          m > k > n in (m,k), (k,n), inout
//      (m,n) TRSM   A(k,n) := A(k,k)^{-1} A(k,n)     n < k     in (k,k), inout
//      (k,n) TRTRI  A(k,k) := A(k,k)^{-1}                      inout (k,k)
//
//  pzlauum, upper (step k, A := U U^H):
//
//      HERK   A(n,n) += A(n,k) A(n,k)^H        n < k     in (n,k), inout (n,n)
//      GEMM   A(n,m) += A(n,k) A(m,k)^H        n < m < k in (n,k), (m,k), inout
//      (n,m) TRMM   A(n,k) := A(n,k) A(k,k)^H        n < k     in (k,k), inout
//      (n,k) LAUUM  A(k,k) := A(k,k) A(k,k)^H                  inout (k,k)
//
//  getri, A := U L (step k: row and column k of the product, X(i,j) needs
//  U(i,l) and L(l,j) for l >= max(i,j) only, see getriimpl.h):
//
//      UL     A(k,k) := U(k,k) L(k,k)                       inout (k,k)
//      GEMM   A(k,k) += A(k,l) A(l,k)           l > k       in (k,l), (l,k),
//                                                           inout (k,k)
//      TRMM   A(k,j) := A(k,j) L(j,j)           j > k       in (j,j), inout
//      (k,j) GEMM   A(k,j) += A(k,l) A(l,j)           l > j > k   in (k,l),
//      (l,j),
//                                                           inout (k,j)
//      TRMM   A(i,k) := U(i,i) A(i,k)           i > k       in (i,i), inout
//      (i,k) GEMM   A(i,k) += A(i,l) A(l,k)           l > i > k   in (i,l),
//      (l,k),
//                                                           inout (i,k)
//
//  The tasks are created in the order of the sequential algorithm; the
//  dependencies (also of a write after earlier reads) keep that order
//  where it matters.
//

#include <algorithm>
#include <ulmblas/lapack/lauum.h>
#include <ulmblas/lapack/getriimpl.h>
#include <ulmblas/lapack/lauumimpl.h>
#include <ulmblas/lapack/tasks/inversetasks.h>
#include <ulmblas/lapack/tasks/sync.h>
#include <ulmblas/lapack/trtri.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trlsm.h>
#include <ulmblas/level3/trumm.h>
#include <ulmblas/level3/trusm.h>

namespace ulmBLAS {
namespace tasks {

template <typename IndexType, typename T>
void
trtriLowerTasks(bool unitDiag, IndexType n, T *A, IndexType incRowA,
                IndexType incColA, IndexType nb, int threads)
{
    const int PRIO_CRIT = 2;
    const int PRIO_TRSM = 1;
    const int PRIO_OTHER = 0;
    (void)PRIO_CRIT;
    (void)PRIO_TRSM;
    (void)PRIO_OTHER;
    (void)threads;

    const IndexType rs = incRowA, cs = incColA;
    const IndexType nt = (n + nb - 1) / nb;
    auto tile = [=](IndexType i, IndexType j) {
	return &A[i * nb * rs + j * nb * cs];
    };
    auto size = [=](IndexType i) { return std::min(nb, n - i * nb); };

    //  tile (i,j) in the depend clauses: its first element
#define ULMBLAS_INV_TILE(i, j) A[(i) * nb * rs + (j) * nb * cs]

#pragma omp parallel num_threads(threads)
#pragma omp single
    {
	for (IndexType k = 0; k < nt; ++k) {
	    const IndexType nk = size(k);
	    T *Akk = tile(k, k);

	    for (IndexType m = k + 1; m < nt; ++m) {
		const IndexType nm = size(m);
		T *Amk = tile(m, k);
//  A(m,k) := -A(m,k) A(k,k)^{-1}, i.e. A(k,k)^T X^T = -A(m,k)^T
#pragma omp task depend(in : ULMBLAS_INV_TILE(k, k))                           \
    depend(inout : ULMBLAS_INV_TILE(m, k)) priority(PRIO_CRIT)
		{
		    taskEnter();
		    trusm(nk, nm, T(-1), false, unitDiag, Akk, cs, rs, Amk, cs,
		          rs);
		    taskLeave();
		}
	    }
	    for (IndexType m = k + 1; m < nt; ++m) {
		const IndexType nm = size(m);
		const T *Amk = tile(m, k);
		for (IndexType j = 0; j < k; ++j) {
		    const IndexType nj = size(j);
		    const T *Akj = tile(k, j);
		    T *Amj = tile(m, j);
#pragma omp task depend(in : ULMBLAS_INV_TILE(m, k), ULMBLAS_INV_TILE(k, j))   \
    depend(inout : ULMBLAS_INV_TILE(m, j)) priority(PRIO_OTHER)
		    {
			taskEnter();
			gemm(nm, nj, nk, T(1), false, Amk, rs, cs, false, Akj,
			     rs, cs, T(1), Amj, rs, cs);
			taskLeave();
		    }
		}
	    }
	    for (IndexType j = 0; j < k; ++j) {
		const IndexType nj = size(j);
		T *Akj = tile(k, j);
#pragma omp task depend(in : ULMBLAS_INV_TILE(k, k))                           \
    depend(inout : ULMBLAS_INV_TILE(k, j)) priority(PRIO_TRSM)
		{
		    taskEnter();
		    trlsm(nk, nj, T(1), false, unitDiag, Akk, rs, cs, Akj, rs,
		          cs);
		    taskLeave();
		}
	    }
#pragma omp task depend(inout : ULMBLAS_INV_TILE(k, k)) priority(PRIO_CRIT)
	    {
		taskEnter();
		trtri(true, unitDiag, nk, Akk, rs, cs);
		taskLeave();
	    }
	}
    }
#undef ULMBLAS_INV_TILE
}

template <typename IndexType, typename T>
void
lauumUpperTasks(IndexType n, T *A, IndexType incRowA, IndexType incColA,
                IndexType nb, int threads)
{
    const int PRIO_CRIT = 1;
    const int PRIO_OTHER = 0;
    (void)PRIO_CRIT;
    (void)PRIO_OTHER;
    (void)threads;

    const IndexType rs = incRowA, cs = incColA;
    const IndexType nt = (n + nb - 1) / nb;
    auto tile = [=](IndexType i, IndexType j) {
	return &A[i * nb * rs + j * nb * cs];
    };
    auto size = [=](IndexType i) { return std::min(nb, n - i * nb); };

    //  tile (i,j) in the depend clauses: its first element
#define ULMBLAS_INV_TILE(i, j) A[(i) * nb * rs + (j) * nb * cs]

#pragma omp parallel num_threads(threads)
#pragma omp single
    {
	for (IndexType k = 0; k < nt; ++k) {
	    const IndexType nk = size(k);
	    const T *Akk = tile(k, k);

	    for (IndexType i = 0; i < k; ++i) {
		const IndexType ni = size(i);
		const T *Aik = tile(i, k);
		T *Aii = tile(i, i);
#pragma omp task depend(in : ULMBLAS_INV_TILE(i, k))                           \
    depend(inout : ULMBLAS_INV_TILE(i, i)) priority(PRIO_OTHER)
		{
		    taskEnter();
		    lauum_impl::herkUpper(ni, nk, Aik, rs, cs, Aii, rs, cs);
		    taskLeave();
		}
		for (IndexType m = i + 1; m < k; ++m) {
		    const IndexType nm = size(m);
		    const T *Amk = tile(m, k);
		    T *Aim = tile(i, m);
#pragma omp task depend(in : ULMBLAS_INV_TILE(i, k), ULMBLAS_INV_TILE(m, k))   \
    depend(inout : ULMBLAS_INV_TILE(i, m)) priority(PRIO_OTHER)
		    {
			taskEnter();
			gemm(ni, nm, nk, T(1), false, Aik, rs, cs, true, Amk,
			     cs, rs, T(1), Aim, rs, cs);
			taskLeave();
		    }
		}
	    }
	    for (IndexType i = 0; i < k; ++i) {
		const IndexType ni = size(i);
		T *Aik = tile(i, k);
//  A(i,k) := A(i,k) A(k,k)^H, i.e. A(i,k)^T := conj(A(k,k)) A(i,k)^T
#pragma omp task depend(in : ULMBLAS_INV_TILE(k, k))                           \
    depend(inout : ULMBLAS_INV_TILE(i, k)) priority(PRIO_CRIT)
		{
		    taskEnter();
		    trumm(nk, ni, T(1), true, false, Akk, rs, cs, Aik, cs, rs);
		    taskLeave();
		}
	    }
	    T *Akkw = tile(k, k);
#pragma omp task depend(inout : ULMBLAS_INV_TILE(k, k)) priority(PRIO_CRIT)
	    {
		taskEnter();
		lauum(false, nk, Akkw, rs, cs);
		taskLeave();
	    }
	}
    }
#undef ULMBLAS_INV_TILE
}

template <typename IndexType, typename T>
void
ulTasks(IndexType n, T *A, IndexType incRowA, IndexType incColA, IndexType nb,
        int threads)
{
    const int PRIO_CRIT = 1;
    const int PRIO_OTHER = 0;
    (void)PRIO_CRIT;
    (void)PRIO_OTHER;
    (void)threads;

    const IndexType rs = incRowA, cs = incColA;
    const IndexType nt = (n + nb - 1) / nb;
    auto tile = [=](IndexType i, IndexType j) {
	return &A[i * nb * rs + j * nb * cs];
    };
    auto size = [=](IndexType i) { return std::min(nb, n - i * nb); };

    //  tile (i,j) in the depend clauses: its first element
#define ULMBLAS_INV_TILE(i, j) A[(i) * nb * rs + (j) * nb * cs]

#pragma omp parallel num_threads(threads)
#pragma omp single
    {
	for (IndexType k = 0; k < nt; ++k) {
	    const IndexType nk = size(k);
	    T *Akk = tile(k, k);
#pragma omp task depend(inout : ULMBLAS_INV_TILE(k, k)) priority(PRIO_CRIT)
	    {
		taskEnter();
		getri_impl::rec(nk, Akk, rs, cs);
		taskLeave();
	    }
	    for (IndexType l = k + 1; l < nt; ++l) {
		const IndexType nl = size(l);
		const T *Akl = tile(k, l);
		const T *Alk = tile(l, k);
#pragma omp task depend(in : ULMBLAS_INV_TILE(k, l), ULMBLAS_INV_TILE(l, k))   \
    depend(inout : ULMBLAS_INV_TILE(k, k)) priority(PRIO_CRIT)
		{
		    taskEnter();
		    gemm(nk, nk, nl, T(1), false, Akl, rs, cs, false, Alk, rs,
		         cs, T(1), Akk, rs, cs);
		    taskLeave();
		}
	    }
	    //  row k
	    for (IndexType j = k + 1; j < nt; ++j) {
		const IndexType nj = size(j);
		const T *Ajj = tile(j, j);
		T *Akj = tile(k, j);
#pragma omp task depend(in : ULMBLAS_INV_TILE(j, j))                           \
    depend(inout : ULMBLAS_INV_TILE(k, j)) priority(PRIO_OTHER)
		{
		    taskEnter();
		    getri_impl::trmmRightLowerUnit(nj, nk, Ajj, rs, cs, Akj, rs,
		                                   cs);
		    taskLeave();
		}
		for (IndexType l = j + 1; l < nt; ++l) {
		    const IndexType nl = size(l);
		    const T *Akl = tile(k, l);
		    const T *Alj = tile(l, j);
#pragma omp task depend(in : ULMBLAS_INV_TILE(k, l), ULMBLAS_INV_TILE(l, j))   \
    depend(inout : ULMBLAS_INV_TILE(k, j)) priority(PRIO_OTHER)
		    {
			taskEnter();
			gemm(nk, nj, nl, T(1), false, Akl, rs, cs, false, Alj,
			     rs, cs, T(1), Akj, rs, cs);
			taskLeave();
		    }
		}
	    }
	    //  column k
	    for (IndexType i = k + 1; i < nt; ++i) {
		const IndexType ni = size(i);
		const T *Aii = tile(i, i);
		T *Aik = tile(i, k);
#pragma omp task depend(in : ULMBLAS_INV_TILE(i, i))                           \
    depend(inout : ULMBLAS_INV_TILE(i, k)) priority(PRIO_OTHER)
		{
		    taskEnter();
		    trumm(ni, nk, T(1), false, false, Aii, rs, cs, Aik, rs, cs);
		    taskLeave();
		}
		for (IndexType l = i + 1; l < nt; ++l) {
		    const IndexType nl = size(l);
		    const T *Ail = tile(i, l);
		    const T *Alk = tile(l, k);
#pragma omp task depend(in : ULMBLAS_INV_TILE(i, l), ULMBLAS_INV_TILE(l, k))   \
    depend(inout : ULMBLAS_INV_TILE(i, k)) priority(PRIO_OTHER)
		    {
			taskEnter();
			gemm(ni, nk, nl, T(1), false, Ail, rs, cs, false, Alk,
			     rs, cs, T(1), Aik, rs, cs);
			taskLeave();
		    }
		}
	    }
	}
    }
#undef ULMBLAS_INV_TILE
}

} // namespace tasks
} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_INVERSETASKS_TCC
