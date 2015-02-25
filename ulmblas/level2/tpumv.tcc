#ifndef ULMBLAS_LEVEL2_TPUMV_TCC
#define ULMBLAS_LEVEL2_TPUMV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/tpumv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tpumv(IndexType    n,
      bool         unitDiag,
      bool         conjA,
      const TA     *A,
      TX           *x,
      IndexType    incX)
{
    if (n==0) {
        return;
    }

    if (!conjA) {
        for (IndexType j=0; j<n; ++j) {
            axpy(j, x[j*incX], A, IndexType(1), x, incX);
            if (!unitDiag) {
                x[j*incX] *= A[j];
            }
            A += j+1;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            acxpy(j, x[j*incX], A, IndexType(1), x, incX);
            if (!unitDiag) {
                x[j*incX] *= conjugate(A[j]);
            }
            A += j+1;
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
tpumv(IndexType    n,
      bool         unitDiag,
      const TA     *A,
      TX           *x,
      IndexType    incX)
{
    tpumv(n, unitDiag, false, A, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPUMV_TCC
