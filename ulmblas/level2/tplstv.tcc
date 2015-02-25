#ifndef ULMBLAS_LEVEL2_TPLSTV_TCC
#define ULMBLAS_LEVEL2_TPLSTV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tplstv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tplstv(IndexType    n,
       bool         unitDiag,
       bool         conjA,
       const TA     *A,
       TX           *x,
       IndexType    incX)
{
    if (n==0) {
        return;
    }

    A += (n+1)*n/2-1;

    if (!conjA) {
        for (IndexType j=n-1; j>=0; --j) {
            IndexType len = n - j;

            x[j*incX] -= dotu(len-1, &A[1], IndexType(1), &x[(j+1)*incX], incX);
            if (!unitDiag) {
                x[j*incX] /= A[0];
            }
            A -= len+1;
        }
    } else {
        for (IndexType j=n-1; j>=0; --j) {
            IndexType len = n - j;

            x[j*incX] -= dotc(len-1, &A[1], IndexType(1), &x[(j+1)*incX], incX);
            if (!unitDiag) {
                x[j*incX] /= conjugate(A[0]);
            }
            A -= len+1;
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
tplstv(IndexType    n,
       bool         unitDiag,
       const TA     *A,
       TX           *x,
       IndexType    incX)
{
    tplstv(n, unitDiag, false, A, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TPLSTV_TCC
