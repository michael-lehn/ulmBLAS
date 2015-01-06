#ifndef ULMBLAS_LEVEL3_SYUMM_TCC
#define ULMBLAS_LEVEL3_SYUMM_TCC 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/syupack.h>
#include <ulmblas/level3/syumm.h>


namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
syumm(IndexType    m,
      IndexType    n,
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
    typedef decltype(Alpha(0)*TA(0)*TB(0))  T;

    const IndexType MC = BlockSize<T>::MC;
    const IndexType NC = BlockSize<T>::NC;

    const IndexType MR = BlockSizeUGemm<T>::MR;
    const IndexType NR = BlockSizeUGemm<T>::NR;

    const IndexType mb = (m+MC-1) / MC;
    const IndexType nb = (n+NC-1) / NC;

    const IndexType mc_ = m % MC;
    const IndexType nc_ = n % NC;

    static MemoryPool<T> memoryPool;

    if (m==0 || n==0 || (alpha==Alpha(0) && (beta==Beta(1)))) {
        return;
    }

    if (alpha==Alpha(0)) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    T  *A_ = memoryPool.allocate(MC*MC+MR);
    T  *B_ = memoryPool.allocate(MC*NC+NR);

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || nc_==0) ? NC : nc_;

        for (IndexType l=0; l<mb; ++l) {
            IndexType kc    = (l!=mb-1 || mc_==0) ? MC   : mc_;
            Beta      beta_ = (l==0) ? beta : Beta(1);

            gepack_B(kc, nc, false,
                     &B[l*MC*incRowB+j*NC*incColB], incRowB, incColB,
                     B_);

            for (IndexType i=0; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || mc_==0) ? MC : mc_;

                if (i<l) {
                    gepack_A(mc, kc, false,
                             &A[i*MC*incRowA+l*MC*incColA], incRowA, incColA,
                             A_);
                } else if (i>l) {
                    gepack_A(mc, kc, false,
                             &A[l*MC*incRowA+i*MC*incColA], incColA, incRowA,
                             A_);
                } else {
                    syupack(mc,
                            &A[i*MC*incRowA+i*MC*incColA], incRowA, incColA,
                            A_);
                }

                mgemm(mc, nc, kc, alpha, A_, B_, beta_,
                      &C[i*MC*incRowC+j*NC*incColC],
                      incRowC, incColC);
            }
        }
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYUMM_TCC
