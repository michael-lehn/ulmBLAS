#ifndef ULMBLAS_LEVEL2_HPUR_H
#define ULMBLAS_LEVEL2_HPUR_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
    void
    hpur(IndexType    n,
         const Alpha  &alpha,
         const TX     *x,
         IndexType    incX,
         TA           *A);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPUR_H

#include <ulmblas/level2/hpur.tcc>
