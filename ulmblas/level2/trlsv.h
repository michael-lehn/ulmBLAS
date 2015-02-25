#ifndef ULMBLAS_LEVEL2_TRLSV_H
#define ULMBLAS_LEVEL2_TRLSV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    trlsv(IndexType    n,
          bool         unitDiag,
          bool         conjA,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          TX           *x,
          IndexType    incX);

template <typename IndexType, typename TA, typename TX>
    void
    trlsv(IndexType    n,
          bool         unitDiag,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          TX           *x,
          IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRLSV_H

#include <ulmblas/level2/trlsv.tcc>
