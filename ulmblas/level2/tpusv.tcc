#ifndef ULMBLAS_LEVEL2_TPUSV_TCC
#define ULMBLAS_LEVEL2_TPUSV_TCC 1

#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/tpusv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tpusv(IndexType    n,
      bool         unitDiag,
      const TA     *A,
      TX           *x,
      IndexType    incX)
{
    if (n==0) {
        return;
    }

    A += (n+1)*n/2-n;
    for (IndexType j=n-1; j>=0; --j) {
        if (!unitDiag) {
            x[j*incX] /= A[j];
        }
        axpy(j, -x[j*incX], A, IndexType(1), x, incX);
        A -= j;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPUSV_TCC
