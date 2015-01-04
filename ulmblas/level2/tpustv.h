#ifndef ULMBLAS_LEVEL2_TPUSTV_H
#define ULMBLAS_LEVEL2_TPUSTV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tpustv(IndexType    n,
           bool         unitDiag,
           const TA     *A,
           TX           *x,
           IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPUSTV_H

#include <ulmblas/level2/tpustv.tcc>
