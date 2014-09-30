#ifndef ULMBLAS_LEVEL3_SYUMM_TCC
#define ULMBLAS_LEVEL3_SYUMM_TCC 1

#include <ulmblas/auxiliary/printmatrix.h>

#include <ulmblas/config/blocksize.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/syupack.h>
#include <ulmblas/level3/mgemm.h>
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

    const IndexType mb = (m+MC-1) / MC;
    const IndexType nb = (n+NC-1) / NC;

    const IndexType _mc = m % MC;
    const IndexType _nc = n % NC;

    static MemoryPool<T> memoryPool;

    if (m==0 || n==0 || (alpha==Alpha(0) && (beta==Beta(1)))) {
        return;
    }

    if (alpha==Alpha(0)) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    T  *_A = memoryPool.allocate(MC*MC);
    T  *_B = memoryPool.allocate(MC*NC);

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (IndexType l=0; l<mb; ++l) {
            IndexType kc    = (l!=mb-1 || _mc==0) ? MC   : _mc;
            Beta      _beta = (l==0) ? beta : Beta(1);

            gepack_B(kc, nc,
                     &B[l*MC*incRowB+j*NC*incColB], incRowB, incColB,
                     _B);

            for (IndexType i=0; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                if (i<l) {
                    gepack_A(mc, kc,
                             &A[i*MC*incRowA+l*MC*incColA], incRowA, incColA,
                             _A);
                } else if (i>l) {
                    gepack_A(mc, kc,
                             &A[l*MC*incRowA+i*MC*incColA], incColA, incRowA,
                             _A);
                } else {
                    syupack(mc,
                            &A[i*MC*incRowA+i*MC*incColA], incRowA, incColA,
                            _A);
                }

                mgemm(mc, nc, kc, alpha, _A, _B, _beta,
                      &C[i*MC*incRowC+j*NC*incColC],
                      incRowC, incColC);
            }
        }
    }

    memoryPool.release(_A);
    memoryPool.release(_B);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_SYUMM_TCC
