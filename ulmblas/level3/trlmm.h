#ifndef ULMBLAS_LEVEL3_TRLMM_H
#define ULMBLAS_LEVEL3_TRLMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB>
    void
    trlmm(IndexType    m,
          IndexType    n,
          const Alpha  &alpha,
          bool         conjA,
          bool         unitDiag,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          TB           *B,
          IndexType    incRowB,
          IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_TRLMM_H

#include <ulmblas/level3/trlmm.tcc>
