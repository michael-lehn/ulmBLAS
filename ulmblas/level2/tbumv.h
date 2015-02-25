#ifndef ULMBLAS_LEVEL2_TBUMV_H
#define ULMBLAS_LEVEL2_TBUMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tbumv(IndexType    n,
          IndexType    k,
          bool         unitDiag,
          bool         conjA,
          const TA     *A,
          IndexType    ldA,
          TX           *x,
          IndexType    incX);

template <typename IndexType, typename TA, typename TX>
    void
    tbumv(IndexType    n,
          IndexType    k,
          bool         unitDiag,
          const TA     *A,
          IndexType    ldA,
          TX           *x,
          IndexType    incX);


} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBUMV_H

#include <ulmblas/level2/tbumv.tcc>
