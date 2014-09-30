#ifndef ULMBLAS_LEVEL3_SYURK_H
#define ULMBLAS_LEVEL3_SYURK_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename Beta,
          typename TC>
    void
    syurk(IndexType    n,
          IndexType    k,
          const Alpha  &alpha,
          const TA     *A,
          IndexType    incRowA,
          IndexType    incColA,
          const Beta   &beta,
          TC           *C,
          IndexType    incRowC,
          IndexType    incColC);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYURK_H

#include <ulmblas/level3/syurk.tcc>
