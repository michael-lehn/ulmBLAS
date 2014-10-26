#ifndef ULMBLAS_LEVEL3_MKERNEL_MTRUMM_H
#define ULMBLAS_LEVEL3_MKERNEL_MTRUMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename TB>
    void
    mtrumm(IndexType    mc,
           IndexType    nc,
           const Alpha  &alpha,
           const T      *A_,
           const T      *B_,
           TB           *B,
           IndexType    incRowB,
           IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MTRUMM_H

#include <ulmblas/level3/mkernel/mtrumm.tcc>
