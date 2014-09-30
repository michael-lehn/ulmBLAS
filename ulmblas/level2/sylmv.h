#ifndef ULMBLAS_LEVEL2_SYLMV_H
#define ULMBLAS_LEVEL2_SYLMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
    void
    sylmv(IndexType    n,
          const Alpha  &alpha,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          const TX     *x,
          IndexType    incX,
          const Beta   &beta,
          TY           *y,
          IndexType    incY);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SYLMV_H

#include <ulmblas/level2/sylmv.tcc>
