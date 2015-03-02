#ifndef ULMBLAS_LEVEL2_HPLR_H
#define ULMBLAS_LEVEL2_HPLR_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
    void
    hplr(IndexType    n,
         const Alpha  &alpha,
         bool         conjX,
         const TX     *x,
         IndexType    incX,
         TA           *A);

template <typename IndexType, typename Alpha, typename TX, typename TA>
    void
    hplr(IndexType    n,
         const Alpha  &alpha,
         const TX     *x,
         IndexType    incX,
         TA           *A);


} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPLR_H

#include <ulmblas/level2/hplr.tcc>
