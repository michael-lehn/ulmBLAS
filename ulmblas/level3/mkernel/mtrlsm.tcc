#ifndef ULMBLAS_LEVEL3_MKERNEL_MTRLSM_TCC
#define ULMBLAS_LEVEL3_MKERNEL_MTRLSM_TCC 1

#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/mkernel/mtrlsm.h>
#include <ulmblas/level3/ukernel/utrlsm.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename TB>
void
mtrlsm(IndexType    mc,
       IndexType    nc,
       const T      &alpha,
       const T      *A_,
       T            *B_,
       TB           *B,
       IndexType    incRowB,
       IndexType    incColB)
{
    const IndexType MR = BlockSizeUGemm<T>::MR;
    const IndexType NR = BlockSizeUGemm<T>::NR;

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType mr_ = mc % MR;
    const IndexType nr_ = nc % NR;

    IndexType mr, nr;
    IndexType kc;

    const T *nextA;
    const T *nextB;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || nr_==0) ? NR : nr_;
        nextB = &B_[j*mc*NR];


        IndexType ia = 0;
        for (IndexType i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || mr_==0) ? MR : mr_;
            kc    = std::min(i*MR, mc-mr);
            nextA = &A_[(ia+i+1)*MR*MR];

            if (i==mp-1) {
                nextA = A_;
                nextB = &B_[(j+1)*mc*NR];
                if (j==np-1) {
                    nextB = B_;
                }
            }

            if (mr==MR && nr==NR) {
                ugemm(kc,
                      T(-1), &A_[ia*MR*MR], &B_[j*mc*NR],
                      alpha,
                      &B_[(j*mc+kc)*NR], NR, IndexType(1),
                      nextA, nextB);

                utrlsm(&A_[(ia*MR+kc)*MR], &B_[(j*mc+kc)*NR],
                       &B_[(j*mc+kc)*NR], NR, IndexType(1));
            } else {

                // Call buffered micro kernels

                ugemm(mr, nr, kc,
                      T(-1), &A_[ia*MR*MR], &B_[j*mc*NR],
                      alpha,
                      &B_[(j*mc+kc)*NR], NR, IndexType(1),
                      nextA, nextB);

                utrlsm(mr, nr,
                       &A_[(ia*MR+kc)*MR], &B_[(j*mc+kc)*NR],
                       &B_[(j*mc+kc)*NR], NR, IndexType(1));
            }
            ia += i+1;
        }
    }
    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || nr_==0) ? NR : nr_;

        gecopy(mc, nr,
               &B_[j*mc*NR], NR, IndexType(1),
               &B[j*NR*incColB], incRowB, incColB);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MTRLSM_TCC
