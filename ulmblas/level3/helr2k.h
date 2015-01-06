#ifndef ULMBLAS_LEVEL3_HELR2K_H
#define ULMBLAS_LEVEL3_HELR2K_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
    void
    helr2k(IndexType    n,
           IndexType    k,
           bool         conj,
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

#endif // ULMBLAS_LEVEL3_HELR2K_H

#include <ulmblas/level3/helr2k.tcc>
