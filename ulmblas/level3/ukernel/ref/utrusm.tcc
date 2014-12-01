#ifndef ULMBLAS_LEVEL3_UKERNEL_REF_UTRUSM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_REF_UTRUSM_TCC 1

#include <iostream>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level3/ukernel/ref/utrusm.h>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename T>
void
utrusm(const T     *A,
       const T     *B,
       T           *C,
       IndexType   incRowC,
       IndexType   incColC)
{
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    T   C_[MR*NR];

    for (IndexType i=0; i<MR; ++i) {
        for (IndexType j=0; j<NR; ++j) {
            C_[i+j*MR] = B[i*NR+j];
        }
    }

    A += MR*(MR-1);
    for (IndexType i=MR-1; i>=0; --i) {
        for (IndexType j=0; j<NR; ++j) {
            C_[i+j*MR] *= A[i];
            for (IndexType l=0; l<i; ++l) {
                C_[l+j*MR] -= A[l]*C_[i+j*MR];
            }
        }
        A -= MR;
    }

    gecopy(MR, NR, C_, IndexType(1), MR, C, incRowC, incColC);
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_REF_UTRUSM_TCC 1
