#ifndef ULMBLAS_LEVEL3_MKERNEL_MSYLRK_TCC
#define ULMBLAS_LEVEL3_MKERNEL_MSYLRK_TCC 1

#include <algorithm>
#include <ulmblas/level3/mkernel/msylrk.h>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/ukernel/usylrk.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename Beta, typename TC>
void
msylrk(IndexType     mc,
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
    constexpr IndexType MR = BlockSizeUGemm<T>::MR;
    constexpr IndexType NR = BlockSizeUGemm<T>::NR;

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

        //  The diagonal of C runs through (0,0).  Start with the first row
        //  panel containing elements on or below it (row j*NR).
        for (IndexType i=(j*NR)/MR; i<mp; ++i) {
            mr    = (i!=mp-1 || mr_==0) ? MR : mr_;
            nextA = &A_[(i+1)*kc*MR];

            if (i==mp-1) {
                nextA = A_;
                nextB = &B_[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = B_;
                }
            }

            //  Unless all elements of the tile are on or below the diagonal
            //  (first row >= last column) the tile contains the diagonal.
            if (i*MR < j*NR+nr-1) {
                usylrk(mr, nr, kc, i*MR, j*NR,
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

#endif // ULMBLAS_LEVEL3_MKERNEL_MSYLRK_TCC
