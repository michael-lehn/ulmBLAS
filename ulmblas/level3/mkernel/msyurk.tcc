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
    const IndexType MR = BlockSizeUGemm<T>::MR;
    const IndexType NR = BlockSizeUGemm<T>::NR;

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType mr_ = mc % MR;
    const IndexType nr_ = nc % NR;

    IndexType mr, nr;

    const T *nextA;
    const T *nextB;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || nr_==0) ? NR : nr_;
        nextB = &B_[j*kc*NR];

        //  The diagonal of C runs through (0,0).  Row panels up to the last
        //  one containing elements on or above it (row j*NR+nr-1).
        for (IndexType i=0; i<mp && i*MR<=j*NR+nr-1; ++i) {
            mr    = (i!=mp-1 || mr_==0) ? MR : mr_;
            nextA = &A_[(i+1)*kc*MR];

            if (i+1==mp || (i+1)*MR>j*NR+nr-1) {
                nextA = A_;
                nextB = &B_[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = B_;
                }
            }

            //  Unless all elements of the tile are on or above the diagonal
            //  (last row <= first column) the tile contains the diagonal.
            if (i*MR+mr-1 > j*NR) {
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
