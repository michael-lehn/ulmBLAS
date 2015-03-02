#ifndef ULMBLAS_LEVEL2_HPLMV_H
#define ULMBLAS_LEVEL2_HPLMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
    void
    hplmv(IndexType    n,
          const Alpha  &alpha,
          bool         conjA,
          const TA     *A,
          const TX     *x,
          IndexType    incX,
          const Beta   &beta,
          TY           *y,
          IndexType    incY);

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
    void
    hplmv(IndexType    n,
          const Alpha  &alpha,
          const TA     *A,
          const TX     *x,
          IndexType    incX,
          const Beta   &beta,
          TY           *y,
          IndexType    incY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPLMV_H

#include <ulmblas/level2/hplmv.tcc>
