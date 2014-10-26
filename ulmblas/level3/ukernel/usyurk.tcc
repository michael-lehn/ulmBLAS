#ifndef ULMBLAS_LEVEL3_UKERNEL_USYURK_TCC
#define ULMBLAS_LEVEL3_UKERNEL_USYURK_TCC 1

#include <ulmblas/auxiliary/printmatrix.h>

#include <ulmblas/level1extensions/truaxpy.h>
#include <ulmblas/level1extensions/truscal.h>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/ukernel/usyurk.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
void
usyurk(IndexType    mr,
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
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    T   C_[MR*NR];

    ugemm(kc, alpha, A, B, T(0), C_, IndexType(1), MR, nextA, nextB);

    if (jc>ic) {
        gescal(jc-ic, nr, beta, C, incRowC, incColC);
        geaxpy(jc-ic, nr, Beta(1), C_, IndexType(1), MR, C, incRowC, incColC);
        truscal(mr-(jc-ic), nr, false, beta,
                &C[(jc-ic)*incRowC], incRowC, incColC);
        truaxpy(mr-(jc-ic), nr, false, Beta(1),
                &C_[jc-ic], IndexType(1), MR,
                &C[(jc-ic)*incRowC], incRowC, incColC);
    } else {
        truscal(mr, nr-(ic-jc), false, beta,
                &C[(ic-jc)*incColC], incRowC, incColC);
        truaxpy(mr, nr-(ic-jc), false, Beta(1),
                &C_[(ic-jc)*MR], IndexType(1), MR,
                &C[(ic-jc)*incColC], incRowC, incColC);
   }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_USYURK_TCC
