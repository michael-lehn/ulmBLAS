#ifndef ULMBLAS_SRC_LEVEL2_TRUMV_H
#define ULMBLAS_SRC_LEVEL2_TRUMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    trumv(IndexType    n,
          bool         unitDiag,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          TX           *x,
          IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL2_TRUMV_H

#include <src/level2/trumv.tcc>
