#ifndef ULMBLAS_LEVEL2_TRUSV_H
#define ULMBLAS_LEVEL2_TRUSV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    trusv(IndexType    n,
          bool         unitDiag,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          TX           *x,
          IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRUSV_H

#include <ulmblas/level2/trusv.tcc>
