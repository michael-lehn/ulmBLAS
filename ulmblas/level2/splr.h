#ifndef ULMBLAS_LEVEL2_SPLR_H
#define ULMBLAS_LEVEL2_SPLR_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
    void
    splr(IndexType    n,
         const Alpha  &alpha,
         const TX     *x,
         IndexType    incX,
         TA           *A);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SPLR_H

#include <ulmblas/level2/splr.tcc>
