#ifndef ULMBLAS_LAPACK_LAUUM_TCC
#define ULMBLAS_LAPACK_LAUUM_TCC 1

#include <ulmblas/lapack/islapacktype.h>
#include <ulmblas/lapack/lauum.h>
#include <ulmblas/lapack/lauumimpl.h>
#include <ulmblas/lapack/threads.h>
#include <ulmblas/lapack/tasks/inversetasks.h>
#include <ulmblas/lapack/tasks/param.h>

namespace ulmBLAS {

template <typename IndexType, typename T>
void
lauum(bool lower, IndexType n, T *A, IndexType incRowA, IndexType incColA)
{
    ULMBLAS_LAPACK_REQUIRE_FLOATING_POINT(T);

    if (n <= 0) {
	return;
    }
    //  lower: with the transposed view U = L^T, element (i,j) of U U^H is
    //  sum_l L(l,i) conj(L(l,j)), stored at (j,i): element (j,i) of L^H L
    const IndexType rs = lower ? incColA : incRowA;
    const IndexType cs = lower ? incRowA : incColA;
#ifdef _OPENMP
    if (!omp_in_parallel()) {
	const tasks::Param p = tasks::inverseParam(long(n), numThreads());
	if (n > IndexType(p.serialN) && p.threads > 1) {
	    tasks::lauumUpperTasks(n, A, rs, cs, IndexType(p.nb), p.threads);
	    return;
	}
    }
#endif
    lauum_impl::rec(n, A, rs, cs);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_LAUUM_TCC
