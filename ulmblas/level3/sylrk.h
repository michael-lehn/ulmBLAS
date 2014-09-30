#ifndef ULMBLAS_LEVEL3_SYLRK_H
#define ULMBLAS_LEVEL3_SYLRK_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename Beta,
         typename TC>
    void
    sylrk(IndexType    n,
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

#endif // ULMBLAS_LEVEL3_SYLRK_H

#include <ulmblas/level3/sylrk.tcc>
