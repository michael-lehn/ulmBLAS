#ifndef ULMBLAS_LEVEL2_HELR_H
#define ULMBLAS_LEVEL2_HELR_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
    void
    helr(IndexType    n,
         const Alpha  &alpha,
         bool         conjX,
         const TX     *x,
         IndexType    incX,
         TA           *A,
         IndexType    incRowA,
         IndexType    incColA);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HELR_H

#include <ulmblas/level2/helr.tcc>
