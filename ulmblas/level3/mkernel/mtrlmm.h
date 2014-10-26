#ifndef ULMBLAS_LEVEL3_MKERNEL_MTRLMM_H
#define ULMBLAS_LEVEL3_MKERNEL_MTRLMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename TB>
    void
    mtrlmm(IndexType    mc,
           IndexType    nc,
           const Alpha  &alpha,
           const T      *A_,
           const T      *B_,
           TB           *B,
           IndexType    incRowB,
           IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MTRLMM_H

#include <ulmblas/level3/mkernel/mtrlmm.tcc>
