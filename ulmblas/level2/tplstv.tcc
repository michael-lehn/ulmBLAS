#ifndef ULMBLAS_LEVEL2_TPLSTV_TCC
#define ULMBLAS_LEVEL2_TPLSTV_TCC 1

#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tplstv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tplstv(IndexType    n,
       bool         unitDiag,
       const TA     *A,
       TX           *x,
       IndexType    incX)
{
    if (n==0) {
        return;
    }

    A += (n+1)*n/2-1;
    for (IndexType j=n-1; j>=0; --j) {
        IndexType len = n - j;

        x[j*incX] -= dotu(len-1, &A[1], IndexType(1), &x[(j+1)*incX], incX);
        if (!unitDiag) {
            x[j*incX] /= A[0];
        }
        A -= len+1;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPLSTV_TCC
