#ifndef ULMBLAS_LEVEL3_UKERNEL_REF_UTRLSM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_REF_UTRLSM_TCC 1

#include <iostream>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level3/ukernel/ref/utrlsm.h>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename T>
void
utrlsm(const T     *A,
       const T     *B,
       T           *C,
       IndexType   incRowC,
       IndexType   incColC)
{
    const IndexType MR = BlockSizeUGemm<T>::MR;
    const IndexType NR = BlockSizeUGemm<T>::NR;

    T   C_[MR*NR];

    for (IndexType i=0; i<MR; ++i) {
        for (IndexType j=0; j<NR; ++j) {
            C_[i+j*MR] = B[i*NR+j];
        }
    }

    for (IndexType i=0; i<MR; ++i) {
        for (IndexType j=0; j<NR; ++j) {
            C_[i+j*MR] *= A[i];
            for (IndexType l=i+1; l<MR; ++l) {
                C_[l+j*MR] -= A[l]*C_[i+j*MR];
            }
        }
        A += MR;
    }

    gecopy(MR, NR, C_, IndexType(1), MR, C, incRowC, incColC);
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_REF_UTRLSM_TCC 1
