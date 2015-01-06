#ifndef ULMBLAS_LEVEL3_TRUSM_TCC
#define ULMBLAS_LEVEL3_TRUSM_TCC 1

#include <ulmblas/auxiliary/printmatrix.h>

#include <type_traits>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/mkernel/mtrusm.h>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/truspack.h>
#include <ulmblas/level3/trusm.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB>
void
trusm(IndexType    m,
      IndexType    n,
      const Alpha  &alpha,
      bool         conjA,
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

    const IndexType MR = BlockSizeUGemm<T>::MR;
    const IndexType NR = BlockSizeUGemm<T>::NR;

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

        for (IndexType i=mb-1; i>=0; --i) {
            IndexType mc  = (i!=mb-1 || mc_==0) ? MC : mc_;
            Alpha  alpha_ = (i==mb-1) ? alpha : Alpha(1);

            gepack_B(mc, nc, false,
                     &B[i*MC*incRowB+j*NC*incColB], incRowB, incColB,
                     B_);

            // std::cerr << "-- trusm ----------" << std::endl;
            // std::cerr << "m = " << m
            //           << ", mb = " << mb
            //           << ", mc_ = " << mc_
            //           << ", mc = " << mc
            //           << ", MC = " << MC
            //           << ", MR = " << MR
            //           << std::endl;
            // printMatrix(mc, mc, &A[i*MC*(incRowA+incColA)], incRowA, incColA);

            truspack(mc, conjA, unitDiag,
                     &A[i*MC*(incRowA+incColA)], incRowA, incColA,
                     A_);

            // printMatrix(MR, (MC*MC/MR), A_, 1, MR);

            mtrusm(mc, nc, alpha_, A_, B_,
                   &B[i*MC*incRowB+j*NC*incColB], incRowB, incColB);

            for (IndexType l=0; l<i; ++l) {
                gepack_A(MC, mc, conjA,
                         &A[l*MC*incRowA+i*MC*incColA], incRowA, incColA,
                         A_);

                mgemm(MC, nc, mc, T(-1), A_, B_, alpha_,
                      &B[l*MC*incRowB+j*NC*incColB], incRowB, incColB);
            }
        }
    }

    memoryPool.release(A_);
    memoryPool.release(B_);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_TRUSM_TCC
