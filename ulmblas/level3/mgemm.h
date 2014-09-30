#ifndef ULMBLAS_LEVEL3_MGEMM_H
#define ULMBLAS_LEVEL3_MGEMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename Beta,
          typename TC>
    void
    mgemm(IndexType     mc,
          IndexType     nc,
          IndexType     kc,
          const Alpha   &alpha,
          const T       *_A,
          const T       *_B,
          const Beta    &beta,
          TC            *C,
          IndexType     incRowC,
          IndexType     incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MGEMM_H

#include <ulmblas/level3/mgemm.tcc>
