#ifndef ULMBLAS_LEVEL3_MKERNEL_MGEMM_H
#define ULMBLAS_LEVEL3_MKERNEL_MGEMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
    void
    mgemm(IndexType     mc,
          IndexType     nc,
          IndexType     kc,
          const T       &alpha,
          const T       *A_,
          const T       *B_,
          const Beta    &beta,
          TC            *C,
          IndexType     incRowC,
          IndexType     incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MGEMM_H

#include <ulmblas/level3/mkernel/mgemm.tcc>
