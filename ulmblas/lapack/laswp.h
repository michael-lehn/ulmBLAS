#ifndef ULMBLAS_LAPACK_LASWP_H
#define ULMBLAS_LAPACK_LASWP_H 1

namespace ulmBLAS {

template <typename IndexType, typename T, typename PivIndex>
    void
    laswp(IndexType    n,
          T            *A,
          IndexType    incRowA,
          IndexType    incColA,
          IndexType    k1,
          IndexType    k2,
          PivIndex     *piv,
          IndexType    incPiv);

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_LASWP_H

#include <ulmblas/lapack/laswp.tcc>
