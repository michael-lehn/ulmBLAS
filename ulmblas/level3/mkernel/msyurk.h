#ifndef ULMBLAS_LEVEL3_MKERNEL_MSYURK_H
#define ULMBLAS_LEVEL3_MKERNEL_MSYURK_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename Beta,
          typename TC>
    void
    msyurk(IndexType     mc,
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

#endif // ULMBLAS_LEVEL3_MKERNEL_MSYURK_H

#include <ulmblas/level3/mkernel/msyurk.tcc>
