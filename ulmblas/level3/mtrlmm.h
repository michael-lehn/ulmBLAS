#ifndef ULMBLAS_LEVEL3_MTRLMM_H
#define ULMBLAS_LEVEL3_MTRLMM_H 1

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename T, typename TB>
    void
    mtrlmm(IndexType    mc,
           IndexType    nc,
           const Alpha  &alpha,
           const T      *_A,
           const T      *_B,
           TB           *B,
           IndexType    incRowB,
           IndexType    incColB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MTRLMM_H

#include <ulmblas/level3/mtrlmm.tcc>
