#ifndef ULMBLAS_LEVEL2_TPLMTV_TCC
#define ULMBLAS_LEVEL2_TPLMTV_TCC 1

#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tplmtv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tplmtv(IndexType    n,
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
            x[j*incX] *= A[0];
        }
        x[j*incX] += dotu(len-1, &A[1], IndexType(1), &x[(j+1)*incX], incX);
        A += len;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPLMTV_TCC
