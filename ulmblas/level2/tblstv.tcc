#ifndef ULMBLAS_LEVEL2_TBLSTV_TCC
#define ULMBLAS_LEVEL2_TBLSTV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tblstv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tblstv(IndexType    n,
       IndexType    k,
       bool         unitDiag,
       bool         conjA,
       const TA     *A,
       IndexType    ldA,
       TX           *x,
       IndexType    incX)
{
    if (n==0) {
        return;
    }

    A += (n-1)*ldA;
    if (!conjA) {
        for (IndexType j=n-1; j>=0; --j) {
            IndexType i0  = 0;
            IndexType i1  = std::min(1+k, n-j);
            IndexType len = std::max(IndexType(0), i1-i0);

            x[j*incX] -= dotu(len-1, &A[1], IndexType(1), &x[(j+1)*incX], incX);
            if (!unitDiag) {
                x[j*incX] /= A[0];
            }
            A -= ldA;
        }
    } else {
        for (IndexType j=n-1; j>=0; --j) {
            IndexType i0  = 0;
            IndexType i1  = std::min(1+k, n-j);
            IndexType len = std::max(IndexType(0), i1-i0);

            x[j*incX] -= dotc(len-1, &A[1], IndexType(1), &x[(j+1)*incX], incX);
            if (!unitDiag) {
                x[j*incX] /= conjugate(A[0]);
            }
            A -= ldA;
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
tblstv(IndexType    n,
       IndexType    k,
       bool         unitDiag,
       const TA     *A,
       IndexType    ldA,
       TX           *x,
       IndexType    incX)
{
    tblstv(n, k, unitDiag, false, A, ldA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBLSTV_TCC
