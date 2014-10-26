#ifndef ULMBLAS_LEVEL3_TRUMM_TCC
#define ULMBLAS_LEVEL3_TRUMM_TCC 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/mkernel/mtrumm.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/trupack.h>
#include <ulmblas/level3/trumm.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB>
void
trumm(IndexType    m,
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

    const IndexType mb = (m+MC-1) / MC;
    const IndexType nb = (n+NC-1) / NC;

    const IndexType mc_ = m % MC;
    const IndexType nc_ = n % NC;

    static MemoryPool<T> memoryPool;

    if (alpha==Alpha(0)) {
        gescal(m, n, Alpha(0), B, incRowB, incColB);
        return;
    }

    T  *A_ = memoryPool.allocate(MC*MC);
    T  *B_ = memoryPool.allocate(MC*NC);

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || nc_==0) ? NC : nc_;

        for (IndexType l=0; l<mb; ++l) {
            IndexType kc = (l!=mb-1 || mc_==0) ? MC   : mc_;

            gepack_B(kc, nc,
                     &B[l*MC*incRowB+j*NC*incColB], incRowB, incColB,
                     B_);

            trupack(kc, unitDiag,
                    &A[l*MC*(incRowA+incColA)], incRowA, incColA,
                    A_);

            mtrumm(kc, nc, alpha, A_, B_,
                   &B[l*MC*incRowB+j*NC*incColB], incRowB, incColB);

            for (IndexType i=0; i<l; ++i) {
                IndexType mc = (i!=mb-1 || mc_==0) ? MC : mc_;

                gepack_A(mc, kc,
                         &A[i*MC*incRowA+l*MC*incColA], incRowA, incColA,
                         A_);

                mgemm(mc, nc, kc, alpha, A_, B_, T(1),
                      &B[i*MC*incRowB+j*NC*incColB], incRowB, incColB);
            }
        }
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_TRUMM_TCC
