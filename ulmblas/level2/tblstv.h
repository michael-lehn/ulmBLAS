#ifndef ULMBLAS_LEVEL2_TBLSTV_H
#define ULMBLAS_LEVEL2_TBLSTV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tblstv(IndexType    n,
           IndexType    k,
           bool         unitDiag,
           const TA     *A,
           IndexType    ldA,
           TX           *x,
           IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBLSTV_H

#include <ulmblas/level2/tblstv.tcc>
