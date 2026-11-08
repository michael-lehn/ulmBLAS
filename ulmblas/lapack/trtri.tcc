#ifndef ULMBLAS_LAPACK_TRTRI_TCC
#define ULMBLAS_LAPACK_TRTRI_TCC 1

#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/trtri.h>
#include <ulmblas/lapack/threads.h>
#include <ulmblas/lapack/tasks/inversetasks.h>
#include <ulmblas/lapack/tasks/param.h>
#include <ulmblas/level3/trlmm.h>
#include <ulmblas/level3/trumm.h>

namespace ulmBLAS {

namespace trtri_impl {

//  lower triangular X := inv(A), unblocked (as LAPACK's ?trti2)
template <typename IndexType, typename T>
void
trti2(bool unitDiag, IndexType n, T *A, IndexType rs, IndexType cs)
{
    for (IndexType j=n-1; j>=0; --j) {
        T ajj = T(-1);
        if (!unitDiag) {
            A[j*(rs+cs)] = T(1)/A[j*(rs+cs)];
            ajj = -A[j*(rs+cs)];
        }
        //  x = A(j+1:n, j) := X(j+1:n, j+1:n) x ajj, bottom up
        T *x = &A[j*cs];
        for (IndexType i=n-1; i>j; --i) {
            T s = unitDiag ? x[i*rs] : A[i*(rs+cs)]*x[i*rs];
            for (IndexType l=j+1; l<i; ++l) {
                s += A[i*rs+l*cs]*x[l*rs];
            }
            x[i*rs] = s*ajj;
        }
    }
}

#ifndef ULMBLAS_TRTRI_BASE
#define ULMBLAS_TRTRI_BASE 32
#endif

//  lower: [X11 0; X21 X22] with X21 = -X22 A21 X11
template <typename IndexType, typename T>
void
rec(bool unitDiag, IndexType n, T *A, IndexType rs, IndexType cs)
{
    if (n<=ULMBLAS_TRTRI_BASE) {
        trti2(unitDiag, n, A, rs, cs);
        return;
    }
    const IndexType n1 = n/2, n2 = n-n1;
    T *A21 = &A[n1*rs];
    T *A22 = &A[n1*(rs+cs)];

    rec(unitDiag, n1, A, rs, cs);
    rec(unitDiag, n2, A22, rs, cs);
    //  A21 := A21 X11, i.e. A21^T := X11^T A21^T (X11^T upper)
    trumm(n1, n2, T(1), false, unitDiag, A, cs, rs, A21, cs, rs);
    //  A21 := -X22 A21
    trlmm(n2, n1, T(-1), false, unitDiag, A22, rs, cs, A21, rs, cs);
}

} // namespace trtri_impl

template <typename IndexType, typename T>
IndexType
trtri(bool         lower,
      bool         unitDiag,
      IndexType    n,
      T            *A,
      IndexType    incRowA,
      IndexType    incColA)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (n<=0) {
        return 0;
    }
    if (!unitDiag) {
        for (IndexType i=0; i<n; ++i) {
            if (A[i*(incRowA+incColA)]==T(0)) {
                return i+1;
            }
        }
    }
    //  upper: the lower triangle of the transposed view (inv(U)^T =
    //  inv(U^T))
    const IndexType rs = lower ? incRowA : incColA;
    const IndexType cs = lower ? incColA : incRowA;
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        const tasks::Param p = tasks::inverseParam(long(n), numThreads());
        if (n>IndexType(p.serialN) && p.threads>1) {
            tasks::trtriLowerTasks(unitDiag, n, A, rs, cs, IndexType(p.nb),
                                   p.threads);
            return 0;
        }
    }
#endif
    trtri_impl::rec(unitDiag, n, A, rs, cs);
    return 0;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_TRTRI_TCC
