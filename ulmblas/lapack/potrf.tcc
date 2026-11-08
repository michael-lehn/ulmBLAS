#ifndef ULMBLAS_LAPACK_POTRF_TCC
#define ULMBLAS_LAPACK_POTRF_TCC 1

#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/lapack/potrf2.h>

#ifdef _OPENMP
#   include <omp.h>
#   include <ulmblas/lapack/threads.h>
#   include <ulmblas/lapack/tasks/param.h>
#   include <ulmblas/lapack/tasks/potrftasks.h>
#endif

namespace ulmBLAS {

template <typename IndexType, typename T>
IndexType
potrf(bool         lower,
      IndexType    n,
      T            *A,
      IndexType    incRowA,
      IndexType    incColA)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (n<=0) {
        return 0;
    }
    //  The upper variant A = U^H U is the lower one of the transposed view
    //  (strides exchanged): U = L^T.  For complex A the transposed view is
    //  conj(A), which is Hermitian too: conj(A) = L L^H gives A = U^H U
    //  with U = L^T.
    const IndexType rs = lower ? incRowA : incColA;
    const IndexType cs = lower ? incColA : incRowA;

#ifdef _OPENMP
    if (!omp_in_parallel()) {
        const tasks::Param p = tasks::potrfParam(long(n), numThreads());
        if (n>IndexType(p.serialN) && p.threads>1 && n>IndexType(p.nb)) {
            return tasks::potrfTasks(n, A, rs, cs, IndexType(p.nb), p.threads);
        }
    }
#endif
    return potrf2(n, A, rs, cs);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRF_TCC
