#ifndef ULMBLAS_LEVEL3_GEMM_H
#define ULMBLAS_LEVEL3_GEMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
    void
    gemm(IndexType    m,
         IndexType    n,
         IndexType    k,
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

#endif // ULMBLAS_LEVEL3_GEMM_H

#include <ulmblas/level3/gemm.tcc>
