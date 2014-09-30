#ifndef ULMBLAS_LEVEL3_TRUMM_H
#define ULMBLAS_LEVEL3_TRUMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB>
    void
    trumm(IndexType    m,
          IndexType    n,
          const Alpha  &alpha,
          bool         unitDiag,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          TB           *B,
          IndexType    incRowB,
          IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_TRUMM_H

#include <ulmblas/level3/trumm.tcc>
