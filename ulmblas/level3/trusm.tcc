#ifndef ULMBLAS_LEVEL3_TRUSM_TCC
#define ULMBLAS_LEVEL3_TRUSM_TCC 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/trupack.h>
#include <ulmblas/level3/mgemm.h>
#include <ulmblas/level3/mtrumm.h>
#include <ulmblas/level3/trumm.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB>
void
trusm(IndexType    m,
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
        for (IndexType i=m-1; i>=0; --i) {
            for (IndexType l=i+1; l<m; ++l) {
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

#endif // ULMBLAS_LEVEL3_TRUSM_TCC
