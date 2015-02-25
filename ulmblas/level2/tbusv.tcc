#ifndef ULMBLAS_LEVEL2_TBUSV_TCC
#define ULMBLAS_LEVEL2_TBUSV_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/tbusv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tbusv(IndexType    n,
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
            IndexType i0  = std::max(IndexType(0), k-j);
            IndexType i1  = k;
            IndexType len = std::max(IndexType(0), i1-i0+1);

            IndexType iX  = std::max(IndexType(0), j-k);

            if (!unitDiag) {
                x[j*incX] /= A[i1];
            }
            axpy(len-1, -x[j*incX], &A[i0], IndexType(1), &x[iX*incX], incX);
            A -= ldA;
        }
    } else {
        for (IndexType j=n-1; j>=0; --j) {
            IndexType i0  = std::max(IndexType(0), k-j);
            IndexType i1  = k;
            IndexType len = std::max(IndexType(0), i1-i0+1);

            IndexType iX  = std::max(IndexType(0), j-k);

            if (!unitDiag) {
                x[j*incX] /= conjugate(A[i1]);
            }
            acxpy(len-1, -x[j*incX], &A[i0], IndexType(1), &x[iX*incX], incX);
            A -= ldA;
        }
    }
}

template <typename IndexType, typename TA, typename TX>
void
tbusv(IndexType    n,
      IndexType    k,
      bool         unitDiag,
      const TA     *A,
      IndexType    ldA,
      TX           *x,
      IndexType    incX)
{
    tbusv(n, k, unitDiag, false, A, ldA, x, incX);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBUSV_TCC
