#ifndef ULMBLAS_LEVEL2_TPUMTV_TCC
#define ULMBLAS_LEVEL2_TPUMTV_TCC 1

#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tpumtv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tpumtv(IndexType    n,
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
            x[j*incX] *= A[j];
        }
        x[j*incX] += dotu(j, A, IndexType(1), x, incX);
        A -= j;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPUMTV_TCC
