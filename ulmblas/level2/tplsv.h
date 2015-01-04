#ifndef ULMBLAS_LEVEL2_TPLSV_H
#define ULMBLAS_LEVEL2_TPLSV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tplsv(IndexType    n,
          bool         unitDiag,
          const TA     *A,
          TX           *x,
          IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPLSV_H

#include <ulmblas/level2/tplsv.tcc>
