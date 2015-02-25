#ifndef ULMBLAS_LEVEL2_TPLSTV_H
#define ULMBLAS_LEVEL2_TPLSTV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tplstv(IndexType    n,
           bool         unitDiag,
           bool         conjA,
           const TA     *A,
           TX           *x,
           IndexType    incX);

template <typename IndexType, typename TA, typename TX>
    void
    tplstv(IndexType    n,
           bool         unitDiag,
           const TA     *A,
           TX           *x,
           IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPLSTV_H

#include <ulmblas/level2/tplstv.tcc>
