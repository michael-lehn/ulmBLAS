#ifndef ULMBLAS_LAPACK_GETRI_TCC
#define ULMBLAS_LAPACK_GETRI_TCC 1

#include <utility>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/getriimpl.h>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/trtri.h>
#include <ulmblas/lapack/unitstride.h>

#ifdef _OPENMP
#include <omp.h>
#include <ulmblas/lapack/threads.h>
#include <ulmblas/lapack/tasks/inversetasks.h>
#include <ulmblas/lapack/tasks/param.h>
#endif

namespace ulmBLAS {

namespace getri_impl {

//  inv(U) inv(L) in place (U non-singular)
template <typename IndexType, typename T>
void
inverse(IndexType n, T *A, IndexType rs, IndexType cs)
{
    trtri(false, false, n, A, rs, cs);
    trtri(true, true, n, A, rs, cs);
#ifdef _OPENMP
    if (!omp_in_parallel()) {
	const tasks::Param p = tasks::inverseParam(long(n), numThreads());
	if (n > IndexType(p.serialN) && p.threads > 1) {
	    tasks::ulTasks(n, A, rs, cs, IndexType(p.nb), p.threads);
	    return;
	}
    }
#endif
    rec(n, A, rs, cs);
}

} // namespace getri_impl

template <typename IndexType, typename T, typename PivIndex>
IndexType
getri(IndexType n, T *A, IndexType incRowA, IndexType incColA,
      const PivIndex *piv, IndexType incPiv, IndexType pivBase)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    const IndexType rs = incRowA, cs = incColA;

    if (n <= 0) {
	return 0;
    }
    for (IndexType i = 0; i < n; ++i) {
	if (A[i * (rs + cs)] == T(0)) {
	    return i + 1;
	}
    }
    //  neither stride 1: on a copy if it fits (lapackCopyFits)
    withUnitStride(n, n, A, rs, cs, false,
                   [n](T *B, IndexType brs, IndexType bcs) {
	               getri_impl::inverse(n, B, brs, bcs);
	               return IndexType(0);
                   });
    //  column interchanges: X P
    for (IndexType j = n - 2; j >= 0; --j) {
	const IndexType jp = IndexType(piv[j * incPiv]) - pivBase;
	if (jp != j) {
	    for (IndexType i = 0; i < n; ++i) {
		std::swap(A[i * rs + j * cs], A[i * rs + jp * cs]);
	    }
	}
    }
    return 0;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRI_TCC
