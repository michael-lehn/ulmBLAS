#ifndef ULMBLAS_LEVEL2_TPUMV_H
#define ULMBLAS_LEVEL2_TPUMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tpumv(IndexType    n,
          bool         unitDiag,
          const TA     *A,
          TX           *x,
          IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPUMV_H

#include <ulmblas/level2/tpumv.tcc>
