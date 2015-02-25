#ifndef ULMBLAS_LEVEL2_HELR2_H
#define ULMBLAS_LEVEL2_HELR2_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
    void
    helr2(IndexType    n,
          bool         conj,
          const Alpha  &alpha,
          const TX     *x,
          IndexType    incX,
          const TY     *y,
          IndexType    incY,
          TA           *A,
          IndexType    incRowA,
          IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HELR2_H

#include <ulmblas/level2/helr2.tcc>
