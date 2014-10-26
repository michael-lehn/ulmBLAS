#ifndef ULMBLAS_LEVEL3_MKERNEL_MSYURK_TCC
#define ULMBLAS_LEVEL3_MKERNEL_MSYURK_TCC 1

#include <algorithm>
#include <ulmblas/level3/mkernel/msyurk.h>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/ukernel/usyurk.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
void
msyurk(IndexType     mc,
       IndexType     nc,
       IndexType     kc,
       const T       &alpha,
       const T       *A_,
       const T       *B_,
       const Beta    &beta,
       TC            *C,
       IndexType     incRowC,
       IndexType     incColC)
{
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    assert((MR%NR==0) || (NR%MR==0));

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType mr_ = mc % MR;
    const IndexType nr_ = nc % NR;

    const IndexType ki = (MR<NR) ? NR/MR : 1;  // 2
    const IndexType kj = (MR>NR) ? MR/NR : 1;  // 1

    IndexType mr, nr;

    const T *nextA;
    const T *nextB;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || nr_==0) ? NR : nr_;
        nextB = &B_[j*kc*NR];

        for (IndexType i=0; (i/ki)<=(j/kj); ++i) {
            mr    = (i!=mp-1 || mr_==0) ? MR : mr_;
            nextA = &A_[(i+1)*kc*MR];

            if (((i/ki)==(j/kj)) && ((i+1)/ki)>(j/kj)) {
                nextA = A_;
                nextB = &B_[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = B_;
                }
            }

            if ((i/ki)==(j/kj)) {
                usyurk(mr, nr, kc, i*MR, j*NR,
                       alpha, &A_[i*kc*MR], &B_[j*kc*NR],
                       beta,
                       &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                       nextA, nextB);
            } else {
                if (mr==MR && nr==NR) {
                    ugemm(kc,
                          alpha, &A_[i*kc*MR], &B_[j*kc*NR],
                          beta,
                          &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                          nextA, nextB);
                } else {
                    // Call the buffered micro kernel
                    ugemm(mr, nr, kc,
                          alpha, &A_[i*kc*MR], &B_[j*kc*NR],
                          beta,
                          &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                          nextA, nextB);
                }
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MSYURK_TCC
