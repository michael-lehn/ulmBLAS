#ifndef ULMBLAS_LAPACK_GETRF_TCC
#define ULMBLAS_LAPACK_GETRF_TCC 1

#include <algorithm>
#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/buffer.h>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getrf2.h>

#ifdef _OPENMP
#include <omp.h>
#include <ulmblas/lapack/threads.h>
#include <ulmblas/lapack/tasks/getrftasks.h>
#include <ulmblas/lapack/tasks/param.h>
#endif

namespace ulmBLAS {

template <typename IndexType, typename T, typename PivIndex>
IndexType
getrf(IndexType m, IndexType n, T *A, IndexType incRowA, IndexType incColA,
      PivIndex *piv, IndexType incPiv)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (m <= 0 || n <= 0) {
	return 0;
    }
#ifdef _OPENMP
    if (!omp_in_parallel()) {
	const IndexType kmin = std::min(m, n);
	const tasks::Param p =
	    tasks::getrfParam(long(m), long(n), numThreads());
	if (kmin > IndexType(p.serialN) && p.threads > 1 &&
	    kmin > IndexType(p.nb)) {
	    Buffer<IndexType> pv(kmin);
	    const IndexType info = tasks::getrfTasks(
	        m, n, A, incRowA, incColA, pv.data(), IndexType(p.nb),
	        IndexType(p.ib), p.panelThreads, p.threads);
	    for (IndexType i = 0; i < kmin; ++i) {
		piv[i * incPiv] = PivIndex(pv[i]);
	    }
	    return info;
	}
    }
#endif
    return getrf2(m, n, A, incRowA, incColA, piv, incPiv);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GETRF_TCC
