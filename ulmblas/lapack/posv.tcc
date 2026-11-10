#ifndef ULMBLAS_LAPACK_POSV_TCC
#define ULMBLAS_LAPACK_POSV_TCC 1

#include <ulmblas/lapack/posv.h>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/lapack/potrs.h>

namespace ulmBLAS {

template <typename IndexType, typename T>
IndexType
posv(bool lower, IndexType n, IndexType nrhs, T *A, IndexType incRowA,
     IndexType incColA, T *B, IndexType incRowB, IndexType incColB)
{
    const IndexType info = potrf(lower, n, A, incRowA, incColA);
    if (info == 0) {
	potrs(lower, n, nrhs, A, incRowA, incColA, B, incRowB, incColB);
    }
    return info;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POSV_TCC
