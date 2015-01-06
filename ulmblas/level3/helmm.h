#ifndef ULMBLAS_LEVEL3_HELMM_H
#define ULMBLAS_LEVEL3_HELMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
    void
    helmm(IndexType    m,
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

#endif // ULMBLAS_LEVEL3_HELMM_H

#include <ulmblas/level3/helmm.tcc>
