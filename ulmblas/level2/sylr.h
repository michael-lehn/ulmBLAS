#ifndef ULMBLAS_LEVEL2_SYLR_H
#define ULMBLAS_LEVEL2_SYLR_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
    void
    sylr(IndexType    n,
         const Alpha  &alpha,
         const TX     *x,
         IndexType    incX,
         TA           *A,
         IndexType    incRowA,
         IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SYLR_H

#include <ulmblas/level2/sylr.tcc>
