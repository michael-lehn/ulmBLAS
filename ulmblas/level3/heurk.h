#ifndef ULMBLAS_LEVEL3_HEURK_H
#define ULMBLAS_LEVEL3_HEURK_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename Beta,
          typename TC>
    void
    heurk(IndexType    n,
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

#endif // ULMBLAS_LEVEL3_HEURK_H

#include <ulmblas/level3/heurk.tcc>
