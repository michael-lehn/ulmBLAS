#ifndef ULMBLAS_LAPACK_GESV_TCC
#define ULMBLAS_LAPACK_GESV_TCC 1

#include <ulmblas/lapack/gesv.h>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getrs.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename PivIndex>
IndexType
gesv(IndexType    n,
     IndexType    nrhs,
     T            *A,
     IndexType    incRowA,
     IndexType    incColA,
     PivIndex     *piv,
     IndexType    incPiv,
     T            *B,
     IndexType    incRowB,
     IndexType    incColB)
{
    const IndexType info = getrf(n, n, A, incRowA, incColA, piv, incPiv);
    if (info==0) {
        getrs(false, false, n, nrhs, A, incRowA, incColA, piv, incPiv,
              B, incRowB, incColB);
    }
    return info;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_GESV_TCC
