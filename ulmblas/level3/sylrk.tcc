#ifndef ULMBLAS_LEVEL3_SYLRK_TCC
#define ULMBLAS_LEVEL3_SYLRK_TCC 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level1extensions/trlscal.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/mgemm.h>
#include <ulmblas/level3/msylrk.h>
#include <ulmblas/level3/sylrk.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename Beta,
          typename TC>
void
sylrk(IndexType    n,
      IndexType    k,
      const Alpha  &alpha,
      const TA     *A,
      IndexType    incRowA,
      IndexType    incColA,
      const Beta   &beta,
      TC           *C,
      IndexType    incRowC,
      IndexType    incColC)
{
    typedef decltype(Alpha(0)*TA(0))  T;

    const IndexType MC = BlockSize<T>::MC;

    const IndexType MR = BlockSize<T>::MC;
    const IndexType NR = BlockSize<T>::NR;

    const IndexType mb = (n+MC-1) / MC;
    const IndexType kb = (k+MC-1) / MC;

    const IndexType _mc = n % MC;
    const IndexType _kc = k % MC;

    static MemoryPool<T> memoryPool;

    if (n==0 || ((alpha==Alpha(0) || k==0) && beta==Beta(1))) {
        return;
    }

    if (alpha==Alpha(0) || k==0) {
        trlscal(n, n, false, beta, C, incRowC, incColC);
        return;
    }

    T  *_A = memoryPool.allocate(MC*MC+MR);
    T  *_B = memoryPool.allocate(MC*MC+NR);

    for (IndexType j=0; j<mb; ++j) {
        IndexType nc = (j!=mb-1 || _mc==0) ? MC : _mc;

        for (IndexType l=0; l<kb; ++l) {
            IndexType kc    = (l!=kb-1 || _kc==0) ? MC   : _kc;
            Beta      _beta = (l==0) ? beta : Beta(1);

            gepack_B(kc, nc,
                     &A[l*MC*incColA+j*MC*incRowA], incColA, incRowA,
                     _B);

            for (IndexType i=j; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                gepack_A(mc, kc,
                         &A[i*MC*incRowA+l*MC*incColA], incRowA, incColA,
                         _A);

                if (i==j) {
                    msylrk(mc, nc, kc, alpha, _A, _B, _beta,
                           &C[i*MC*incRowC+j*MC*incColC],
                           incRowC, incColC);
                } else {
                    mgemm(mc, nc, kc, alpha, _A, _B, _beta,
                          &C[i*MC*incRowC+j*MC*incColC],
                          incRowC, incColC);
                }
            }
        }
    }

    memoryPool.release(_A);
    memoryPool.release(_B);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYLRK_TCC
