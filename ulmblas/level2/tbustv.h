#ifndef ULMBLAS_LEVEL2_TBUSTV_H
#define ULMBLAS_LEVEL2_TBUSTV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tbustv(IndexType    n,
           IndexType    k,
           bool         unitDiag,
           bool         conjA,
           const TA     *A,
           IndexType    ldA,
           const TX     *x,
           IndexType    incX);

template <typename IndexType, typename TA, typename TX>
    void
    tbustv(IndexType    n,
           IndexType    k,
           bool         unitDiag,
           const TA     *A,
           IndexType    ldA,
           const TX     *x,
           IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBUSTV_H

#include <ulmblas/level2/tbustv.tcc>
