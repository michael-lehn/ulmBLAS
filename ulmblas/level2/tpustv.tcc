#ifndef ULMBLAS_LEVEL2_TPUSTV_TCC
#define ULMBLAS_LEVEL2_TPUSTV_TCC 1

#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tpustv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tpustv(IndexType    n,
       bool         unitDiag,
       const TA     *A,
       TX           *x,
       IndexType    incX)
{
    if (n==0) {
        return;
    }

    for (IndexType j=0; j<n; ++j) {
        x[j*incX] -= dotu(j, A, IndexType(1), x, incX);
        if (!unitDiag) {
            x[j*incX] /= A[j];
        }
        A += j+1;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPUSTV_TCC
