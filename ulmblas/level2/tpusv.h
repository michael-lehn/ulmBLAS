#ifndef ULMBLAS_LEVEL2_TPUSV_H
#define ULMBLAS_LEVEL2_TPUSV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tpusv(IndexType    n,
          bool         unitDiag,
          bool         conjA,
          const TA     *A,
          TX           *x,
          IndexType    incX);

template <typename IndexType, typename TA, typename TX>
    void
    tpusv(IndexType    n,
          bool         unitDiag,
          const TA     *A,
          TX           *x,
          IndexType    incX);


} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPUSV_H

#include <ulmblas/level2/tpusv.tcc>
