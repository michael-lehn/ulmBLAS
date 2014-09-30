#ifndef ULMBLAS_LEVEL2_TRLMV_H
#define ULMBLAS_LEVEL2_TRLMV_H 1

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
    void
    trlmv(IndexType    n,
          bool         unitDiag,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          TX           *x,
          IndexType    incX);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TRLMV_H

#include <ulmblas/level2/trlmv.tcc>
