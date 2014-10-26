#ifndef ULMBLAS_LEVEL3_MKERNEL_MTRLSM_H
#define ULMBLAS_LEVEL3_MKERNEL_MTRLSM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename TB>
    void
    mtrlsm(IndexType    mc,
           IndexType    nc,
           const Alpha  &alpha,
           const T      *A_,
           T            *B_,
           TB           *B,
           IndexType    incRowB,
           IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MTRLSM_H

#include <ulmblas/level3/mkernel/mtrlsm.tcc>
