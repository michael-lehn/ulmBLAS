#ifndef ULMBLAS_LEVEL3_USYLRK_H
#define ULMBLAS_LEVEL3_USYLRK_H 1

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
    void
    usylrk(IndexType    mr,
           IndexType    nr,
           IndexType    kc,
           const T      &alpha,
           const T      *A,
           const T      *B,
           const Beta   &beta,
           TC           *C,
           IndexType    incRowC,
           IndexType    incColC,
           const T      *nextA,
           const T      *nextB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_USYLRK_H

#include <ulmblas/level3/usylrk.tcc>
