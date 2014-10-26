#ifndef ULMBLAS_LEVEL3_TRLSM_TCC
#define ULMBLAS_LEVEL3_TRLSM_TCC 1

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/mkernel/mtrlsm.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/trlspack.h>
#include <ulmblas/level3/trlsm.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB>
void
trlsm(IndexType    m,
      IndexType    n,
      const Alpha  &alpha,
      bool         unitDiag,
      const TA     *A,
      IndexType    incRowA,
      IndexType    incColA,
      TB           *B,
      IndexType    incRowB,
      IndexType    incColB)
{
    typedef decltype(Alpha(0)*TA(0)*TB(0))  T;

    const IndexType MC = BlockSize<T>::MC;
    const IndexType NC = BlockSize<T>::NC;

    const IndexType MR = BlockSize<T>::MR;
    const IndexType NR = BlockSize<T>::NR;

    const IndexType mb = (m+MC-1) / MC;
    const IndexType nb = (n+NC-1) / NC;

    const IndexType mc_ = m % MC;
    const IndexType nc_ = n % NC;

    static MemoryPool<T> memoryPool;

    if (alpha==Alpha(0)) {
        gescal(m, n, Alpha(0), B, incRowB, incColB);
        return;
    }

    T  *A_ = memoryPool.allocate(MC*MC+MR);
    T  *B_ = memoryPool.allocate(MC*NC+NR);

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || nc_==0) ? NC : nc_;

        for (IndexType i=0; i<mb; ++i) {
            IndexType mc  = (i!=mb-1 || mc_==0) ? MC : mc_;
            Alpha  alpha_ = (i==0) ? alpha : Alpha(1);

            gepack_B(mc, nc,
                     &B[i*MC*incRowB+j*NC*incColB], incRowB, incColB,
                     B_);

            trlspack(mc, unitDiag,
                     &A[i*MC*(incRowA+incColA)], incRowA, incColA,
                     A_);

            mtrlsm(mc, nc, alpha_, A_, B_,
                   &B[i*MC*incRowB+j*NC*incColB], incRowB, incColB);

            for (IndexType l=i+1; l<mb; ++l) {
                mc  = (l!=mb-1 || mc_==0) ? MC : mc_;

                gepack_A(mc, MC,
                         &A[l*MC*incRowA+i*MC*incColA], incRowA, incColA,
                         A_);

                mgemm(mc, nc, MC, T(-1), A_, B_, alpha_,
                      &B[l*MC*incRowB+j*NC*incColB], incRowB, incColB);
            }
        }
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_TRLSM_TCC
