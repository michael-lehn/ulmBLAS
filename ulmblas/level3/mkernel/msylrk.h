#ifndef ULMBLAS_LEVEL3_MKERNEL_MSYLRK_H
#define ULMBLAS_LEVEL3_MKERNEL_MSYLRK_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename Beta,
          typename TC>
    void
    msylrk(IndexType     mc,
           IndexType     nc,
           IndexType     kc,
           const Alpha   &alpha,
           const T       *A_,
           const T       *B_,
           const Beta    &beta,
           TC            *C,
           IndexType     incRowC,
           IndexType     incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MSYLRK_H

#include <ulmblas/level3/mkernel/msylrk.tcc>
