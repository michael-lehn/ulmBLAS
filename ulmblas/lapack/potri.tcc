#ifndef ULMBLAS_LAPACK_POTRI_TCC
#define ULMBLAS_LAPACK_POTRI_TCC 1

#include <ulmblas/lapack/lauum.h>
#include <ulmblas/lapack/potri.h>
#include <ulmblas/lapack/trtri.h>

namespace ulmBLAS {

template <typename IndexType, typename T>
IndexType
potri(bool         lower,
      IndexType    n,
      T            *A,
      IndexType    incRowA,
      IndexType    incColA)
{
    const IndexType info = trtri(lower, false, n, A, incRowA, incColA);
    if (info==0) {
        lauum(lower, n, A, incRowA, incColA);
    }
    return info;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_POTRI_TCC
