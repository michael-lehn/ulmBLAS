#ifndef ULMBLAS_LEVEL2_TBUMV_TCC
#define ULMBLAS_LEVEL2_TBUMV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/tbumv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tbumv(IndexType    n,
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
            IndexType i1  = k;
            IndexType len = std::max(IndexType(0), i1-i0+1);

            IndexType iX  = std::max(IndexType(0), j-k);

            axpy(len-1, x[j*incX], &A[i0], IndexType(1), &x[iX*incX], incX);
            if (!unitDiag) {
                x[j*incX] *= A[i1];
            }
            A += ldA;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            IndexType i0  = std::max(IndexType(0), k-j);
            IndexType i1  = k;
            IndexType len = std::max(IndexType(0), i1-i0+1);

            IndexType iX  = std::max(IndexType(0), j-k);

            acxpy(len-1, x[j*incX], &A[i0], IndexType(1), &x[iX*incX], incX);
            if (!unitDiag) {
                x[j*incX] *= conjugate(A[i1]);
            }
            A += ldA;
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
tbumv(IndexType    n,
      IndexType    k,
      bool         unitDiag,
      const TA     *A,
      IndexType    ldA,
      TX           *x,
      IndexType    incX)
{
    tbumv(n, k, unitDiag, false, A, ldA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBUMV_TCC
