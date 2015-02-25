#ifndef ULMBLAS_LEVEL2_TBUSTV_TCC
#define ULMBLAS_LEVEL2_TBUSTV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tbumtv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tbustv(IndexType    n,
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

    if (!conjA) {
        for (IndexType j=0; j<n; ++j) {
            IndexType i0  = std::max(IndexType(0), k-j);
            IndexType i1  = std::min(k, n+k-1-j);
            IndexType len = std::max(IndexType(0), i1-i0+1);

            IndexType iX  = std::max(IndexType(0), j-k);

            x[j*incX] -= dotu(len-1, &A[i0], IndexType(1), &x[iX*incX], incX);
            if (!unitDiag) {
                x[j*incX] /= A[i1];
            }
            A += ldA;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            IndexType i0  = std::max(IndexType(0), k-j);
            IndexType i1  = std::min(k, n+k-1-j);
            IndexType len = std::max(IndexType(0), i1-i0+1);

            IndexType iX  = std::max(IndexType(0), j-k);

            x[j*incX] -= dotc(len-1, &A[i0], IndexType(1), &x[iX*incX], incX);
            if (!unitDiag) {
                x[j*incX] /= conjugate(A[i1]);
            }
            A += ldA;
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
tbustv(IndexType    n,
       IndexType    k,
       bool         unitDiag,
       const TA     *A,
       IndexType    ldA,
       TX           *x,
       IndexType    incX)
{
    tbustv(n, k, unitDiag, false, A, ldA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBUSTV_TCC
