#ifndef ULMBLAS_LEVEL2_HBLMV_H
#define ULMBLAS_LEVEL2_HBLMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
    void
    hblmv(IndexType    n,
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
    hblmv(IndexType    n,
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

#endif // ULMBLAS_LEVEL2_HBLMV_H

#include <ulmblas/level2/hblmv.tcc>
