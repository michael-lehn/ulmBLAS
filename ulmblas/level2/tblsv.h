#ifndef ULMBLAS_LEVEL2_TBLSV_H
#define ULMBLAS_LEVEL2_TBLSV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    tblsv(IndexType    n,
          IndexType    k,
          bool         unitDiag,
          const TA     *A,
          IndexType    ldA,
          TX           *x,
          IndexType    incX);

template <typename IndexType, typename TA, typename TX>
    void
    tblsv(IndexType    n,
          IndexType    k,
          bool         unitDiag,
          bool         conjA,
          const TA     *A,
          IndexType    ldA,
          TX           *x,
          IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBLSV_H

#include <ulmblas/level2/tblsv.tcc>
