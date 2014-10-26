#ifndef ULMBLAS_LEVEL3_MKERNEL_MTRUSM_TCC
#define ULMBLAS_LEVEL3_MKERNEL_MTRUSM_TCC 1

#include <ulmblas/level3/mkernel/mtrusm.h>
#include <ulmblas/level3/ukernel/utrusm.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename TB>
void
mtrusm(IndexType    mc,
       IndexType    nc,
       const T      &alpha,
       const T      *A_,
       T            *B_,
       TB           *B,
       IndexType    incRowB,
       IndexType    incColB)
{
    const IndexType MR = ugemm_mr<T>();
    const IndexType NR = ugemm_nr<T>();

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType mr_ = mc % MR;
    const IndexType nr_ = nc % NR;

    IndexType mr, nr;
    IndexType kc;

    const T *nextA;
    const T *nextB;

    const IndexType na = mp*(2*mc-(mp-1)*MR)/2;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || nr_==0) ? NR : nr_;
        nextB = &B_[j*mc*NR];


        IndexType ia = na;
        IndexType ib = mc;

        for (IndexType i=mp-1; i>=0; --i) {
            mr    = (i!=mp-1 || mr_==0) ? MR : mr_;
            kc    = std::max(mc-(i+1)*MR, IndexType(0));

            ia    -= mr + kc;
            ib    -= mr;
            nextA = &A_[(ia-1)*MR];

            if (i==0) {
                nextA = A_;
                nextB = &B_[(j+1)*mc*NR];
                if (j==np-1) {
                    nextB = B_;
                }
            }

            if (mr==MR && nr==NR) {
                ugemm(kc,
                      T(-1), &A_[(ia+MR)*MR], &B_[(j*mc+ib+MR)*NR],
                      alpha,
                      &B_[(j*mc+ib)*NR], NR, IndexType(1),
                      nextA, nextB);

                utrusm(&A_[ia*MR], &B_[(j*mc+ib)*NR],
                       &B_[(j*mc+ib)*NR], NR, IndexType(1));
            } else {

                // Call buffered micro kernels

                ugemm(mr, nr, kc,
                      T(-1), &A_[(ia+MR)*MR], &B_[(j*mc+ib+mr)*NR],
                      alpha,
                      &B_[(j*mc+ib)*NR], NR, IndexType(1),
                      nextA, nextB);

                utrusm(mr, nr,
                       &A_[ia*MR], &B_[(j*mc+ib)*NR],
                       &B_[(j*mc+ib)*NR], NR, IndexType(1));
            }
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

#endif // ULMBLAS_LEVEL3_MKERNEL_MTRUSM_TCC
