#ifndef ULMBLAS_LEVEL2_HPLR2_H
#define ULMBLAS_LEVEL2_HPLR2_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
    void
    hplr2(IndexType    n,
          const Alpha  &alpha,
          const TX     *x,
          IndexType    incX,
          const TY     *y,
          IndexType    incY,
          TA           *A);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPLR2_H

#include <ulmblas/level2/hplr2.tcc>
