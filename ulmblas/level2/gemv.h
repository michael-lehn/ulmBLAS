#ifndef ULMBLAS_LEVEL2_GEMV_H
#define ULMBLAS_LEVEL2_GEMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
    void
    gemv(IndexType    m,
         IndexType    n,
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

#endif // ULMBLAS_LEVEL2_GEMV_H

#include <ulmblas/level2/gemv.tcc>
