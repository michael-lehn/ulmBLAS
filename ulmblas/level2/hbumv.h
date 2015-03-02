#ifndef ULMBLAS_LEVEL2_HBUMV_H
#define ULMBLAS_LEVEL2_HBUMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
    void
    hbumv(IndexType    n,
          IndexType    k,
          const Alpha  &alpha,
          bool         conjA,
          const TA     *A,
          IndexType    ldA,
          const TX     *x,
          IndexType    incX,
          const Beta   &beta,
          TY           *y,
          IndexType    incY);

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
    void
    hbumv(IndexType    n,
          IndexType    k,
          const Alpha  &alpha,
          const TA     *A,
          IndexType    ldA,
          const TX     *x,
          IndexType    incX,
          const Beta   &beta,
          TY           *y,
          IndexType    incY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HBUMV_H

#include <ulmblas/level2/hbumv.tcc>
