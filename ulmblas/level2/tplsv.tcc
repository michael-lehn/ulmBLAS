#ifndef ULMBLAS_LEVEL2_TPLSV_TCC
#define ULMBLAS_LEVEL2_TPLSV_TCC 1

#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/tplsv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tplsv(IndexType    n,
      bool         unitDiag,
      const TA     *A,
      TX           *x,
      IndexType    incX)
{
    if (n==0) {
        return;
    }

    for (IndexType j=0; j<n; ++j) {
        IndexType len = n-j;

        if (!unitDiag) {
            x[j*incX] /= A[0];
        }
        axpy(len-1, -x[j*incX], &A[1], IndexType(1), &x[(j+1)*incX], incX);
        A += len;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPLSV_TCC
