#ifndef ULMBLAS_LEVEL3_GEMM_TCC
#define ULMBLAS_LEVEL3_GEMM_TCC 1

#include <stdio.h>

#include <ulmblas/config/blocksize.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/mgemm.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
gemm(IndexType    m,
     IndexType    n,
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
    typedef decltype(Alpha(0)*TA(0)*TB(0))  T;

    const IndexType MC = BlockSize<T>::MC;
    const IndexType NC = BlockSize<T>::NC;
    const IndexType KC = BlockSize<T>::KC;

    const IndexType mb = (m+MC-1) / MC;
    const IndexType nb = (n+NC-1) / NC;
    const IndexType kb = (k+KC-1) / KC;

    const IndexType _mc = m % MC;
    const IndexType _nc = n % NC;
    const IndexType _kc = k % KC;

    static MemoryPool<T> memoryPool;

    if (alpha==Alpha(0) || k==0) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    T  *_A = memoryPool.allocate(MC*KC);
    T  *_B = memoryPool.allocate(KC*NC);

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (IndexType l=0; l<kb; ++l) {
            IndexType kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            Beta      _beta = (l==0) ? beta : Beta(1);

            gepack_B(kc, nc,
                     &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                     _B);

            for (IndexType i=0; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                gepack_A(mc, kc,
                         &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                         _A);

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

#endif // ULMBLAS_LEVEL3_GEMM_TCC
