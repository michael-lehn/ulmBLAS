#ifndef ULMBLAS_LEVEL3_TRLSM_TCC
#define ULMBLAS_LEVEL3_TRLSM_TCC 1

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/trlpack.h>
#include <ulmblas/level3/mgemm.h>
#include <ulmblas/level3/mtrlmm.h>
#include <ulmblas/level3/trlmm.h>

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
    for (IndexType j=0; j<n; ++j) {
        for (IndexType i=0; i<m; ++i) {
            B[i*incRowB+j*incColB] *= alpha;
        }
    }
    for (IndexType j=0; j<n; ++j) {
        for (IndexType i=0; i<m; ++i) {
            for (IndexType l=0; l<i; ++l) {
                B[i*incRowB+j*incColB] -= A[i*incRowA+l*incColA]
                                         *B[l*incRowB+j*incColB];
            }
            if (!unitDiag) {
                B[i*incRowB+j*incColB] /= A[i*incRowA+i*incColA];
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_TRLSM_TCC
