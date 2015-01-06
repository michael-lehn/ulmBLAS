#ifndef ULMBLAS_LEVEL3_SYUR2K_TCC
#define ULMBLAS_LEVEL3_SYUR2K_TCC 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level1extensions/truscal.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/mkernel/msyurk.h>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/syur2k.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
syur2k(IndexType    n,
       IndexType    k,
       const Alpha  &alpha,
       const TA     *A,
       IndexType    incRowA,
       IndexType    incColA,
       const TB     *B,
       IndexType    incRowB,
       IndexType    incColB,
       const Beta   &beta,
       TC           *C,
       IndexType    incRowC,
       IndexType    incColC)
{
    typedef decltype(Alpha(0)*TA(0))  T;

    const IndexType MC = BlockSize<T>::MC;

    const IndexType MR = BlockSizeUGemm<T>::MR;
    const IndexType NR = BlockSizeUGemm<T>::NR;

    const IndexType mb = (n+MC-1) / MC;
    const IndexType kb = (k+MC-1) / MC;

    const IndexType mc_ = n % MC;
    const IndexType kc_ = k % MC;

    static MemoryPool<T> memoryPool;

    if (n==0 || ((alpha==Alpha(0) || k==0) && beta==Beta(1))) {
        return;
    }

    if (alpha==Alpha(0) || k==0) {
        truscal(n, n, false, beta, C, incRowC, incColC);
        return;
    }

    T     *A_ = memoryPool.allocate(MC*MC+MR);
    T     *B_ = memoryPool.allocate(MC*MC+NR);

    Beta  beta_ = Beta(1);

    for (IndexType j=0; j<mb; ++j) {
        IndexType nc = (j!=mb-1 || mc_==0) ? MC : mc_;

        for (IndexType l=0; l<kb; ++l) {
            IndexType kc    = (l!=kb-1 || kc_==0) ? MC   : kc_;
            Beta      beta_ = (l==0) ? beta : Beta(1);

            gepack_B(kc, nc, false,
                     &B[l*MC*incColA+j*MC*incRowA], incColA, incRowA,
                     B_);

            for (IndexType i=0; i<=j; ++i) {
                IndexType mc = (i!=mb-1 || mc_==0) ? MC : mc_;

                gepack_A(mc, kc, false,
                         &A[i*MC*incRowA+l*MC*incColA], incRowA, incColA,
                         A_);

                if (i==j) {
                    msyurk(mc, nc, kc, alpha, A_, B_, beta_,
                           &C[i*MC*incRowC+j*MC*incColC],
                           incRowC, incColC);
                } else {
                    mgemm(mc, nc, kc, alpha, A_, B_, beta_,
                          &C[i*MC*incRowC+j*MC*incColC],
                          incRowC, incColC);
                }
            }
        }
    }

    beta_ = Beta(1);

    for (IndexType j=0; j<mb; ++j) {
        IndexType nc = (j!=mb-1 || mc_==0) ? MC : mc_;

        for (IndexType l=0; l<kb; ++l) {
            IndexType kc    = (l!=kb-1 || kc_==0) ? MC   : kc_;

            gepack_B(kc, nc, false,
                     &A[l*MC*incColA+j*MC*incRowA], incColA, incRowA,
                     B_);

            for (IndexType i=0; i<=j; ++i) {
                IndexType mc = (i!=mb-1 || mc_==0) ? MC : mc_;

                gepack_A(mc, kc, false,
                         &B[i*MC*incRowA+l*MC*incColA], incRowA, incColA,
                         A_);

                if (i==j) {
                    msyurk(mc, nc, kc, alpha, A_, B_, beta_,
                           &C[i*MC*incRowC+j*MC*incColC],
                           incRowC, incColC);
                } else {
                    mgemm(mc, nc, kc, alpha, A_, B_, beta_,
                          &C[i*MC*incRowC+j*MC*incColC],
                          incRowC, incColC);
                }
            }
        }
    }

    memoryPool.release(A_);
    memoryPool.release(B_);

}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYUR2K_TCC
