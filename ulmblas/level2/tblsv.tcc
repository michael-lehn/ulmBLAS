#ifndef ULMBLAS_LEVEL2_TBLSV_TCC
#define ULMBLAS_LEVEL2_TBLSV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/tblsv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tblsv(IndexType    n,
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
            IndexType i1  = k - std::max(IndexType(0), (j+1+k)-n);
            IndexType len = std::max(IndexType(0), i1+1);

            if (!unitDiag) {
                x[j*incX] /= A[0];
            }
            axpy(len-1, -x[j*incX], &A[1], IndexType(1), &x[(j+1)*incX], incX);
            A += ldA;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            IndexType i1  = k - std::max(IndexType(0), (j+1+k)-n);
            IndexType len = std::max(IndexType(0), i1+1);

            if (!unitDiag) {
                x[j*incX] /= conjugate(A[0]);
            }
            acxpy(len-1, -x[j*incX], &A[1], IndexType(1), &x[(j+1)*incX], incX);
            A += ldA;
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
tblsv(IndexType    n,
      IndexType    k,
      bool         unitDiag,
      const TA     *A,
      IndexType    ldA,
      TX           *x,
      IndexType    incX)
{
    tblsv(n, k, unitDiag, false, A, ldA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBLSV_TCC
