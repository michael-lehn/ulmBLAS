#ifndef ULMBLAS_LEVEL3_HEUMM_H
#define ULMBLAS_LEVEL3_HEUMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
    void
    heumm(IndexType    m,
          IndexType    n,
          const Alpha  &alpha,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          const TB     *B,
          IndexType    incRowB,
          IndexType    incColB,
          const Beta   &beta,
          TC           *C,
          IndexType    incRowC,
          IndexType    incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_HEUMM_H

#include <ulmblas/level3/heumm.tcc>
