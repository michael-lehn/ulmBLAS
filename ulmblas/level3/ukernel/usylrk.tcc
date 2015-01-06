#ifndef ULMBLAS_LEVEL3_UKERNEL_USYLRK_TCC
#define ULMBLAS_LEVEL3_UKERNEL_USYLRK_TCC 1

#include <ulmblas/auxiliary/printmatrix.h>

#include <ulmblas/level1extensions/trlaxpy.h>
#include <ulmblas/level1extensions/trlscal.h>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/ukernel/usylrk.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
void
usylrk(IndexType    mr,
       IndexType    nr,
       IndexType    kc,
       IndexType    ic,
       IndexType    jc,
       const T      &alpha,
       const T      *A,
       const T      *B,
       const Beta   &beta,
       TC           *C,
       IndexType    incRowC,
       IndexType    incColC,
       const T      *nextA,
       const T      *nextB)
{
    const IndexType MR = BlockSizeUGemm<T>::MR;
    const IndexType NR = BlockSizeUGemm<T>::NR;

    T   C_[MR*NR];

    ugemm(kc, alpha, A, B, T(0), C_, IndexType(1), MR, nextA, nextB);

    if (jc<ic) {
        gescal(mr, ic-jc, beta, C, incRowC, incColC);
        geaxpy(mr, ic-jc, Beta(1), C_, IndexType(1), MR, C, incRowC, incColC);
        trlscal(mr, nr-(ic-jc), false, beta,
                &C[(ic-jc)*incColC], incRowC, incColC);
        trlaxpy(mr, nr-(ic-jc), false, Beta(1),
                &C_[(ic-jc)*MR], IndexType(1), MR,
                &C[(ic-jc)*incColC], incRowC, incColC);
    } else {
        trlscal(mr-(jc-ic), nr, false, beta,
                &C[(jc-ic)*incRowC], incRowC, incColC);
        trlaxpy(mr-(jc-ic), nr, false, Beta(1),
                &C_[jc-ic], IndexType(1), MR,
                &C[(jc-ic)*incRowC], incRowC, incColC);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_USYLRK_TCC
