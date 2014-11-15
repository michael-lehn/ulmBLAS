#ifndef ULMBLAS_LEVEL3_GEMM_TCC
#define ULMBLAS_LEVEL3_GEMM_TCC 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/pack/gepack.h>

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

    const IndexType mc_ = m % MC;
    const IndexType nc_ = n % NC;
    const IndexType kc_ = k % KC;

    static MemoryPool<T> memoryPool;

    if (alpha==Alpha(0) || k==0) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    T  *A_ = memoryPool.allocate(MC*KC);
    T  *B_ = memoryPool.allocate(KC*NC);

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || nc_==0) ? NC : nc_;

        for (IndexType l=0; l<kb; ++l) {
            IndexType kc    = (l!=kb-1 || kc_==0) ? KC   : kc_;
            Beta      beta_ = (l==0) ? beta : Beta(1);

            gepack_B(kc, nc,
                     &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                     B_);

            for (IndexType i=0; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || mc_==0) ? MC : mc_;

                gepack_A(mc, kc,
                         &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                         A_);

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

#endif // ULMBLAS_LEVEL3_GEMM_TCC
