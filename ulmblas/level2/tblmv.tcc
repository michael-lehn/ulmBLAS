#ifndef ULMBLAS_LEVEL2_TBLMV_TCC
#define ULMBLAS_LEVEL2_TBLMV_TCC 1

#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/tblmv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tblmv(IndexType    n,
      IndexType    k,
      bool         unitDiag,
      const TA     *A,
      IndexType    ldA,
      TX           *x,
      IndexType    incX)
{
    if (n==0) {
        return;
    }

    A += (n-1)*ldA;
    for (IndexType j=n-1; j>=0; --j) {
        IndexType i1  = k - std::max(IndexType(0), (j+1+k)-n);
        IndexType len = std::max(IndexType(0), i1+1);

        axpy(len-1, x[j*incX], &A[1], IndexType(1), &x[(j+1)*incX], incX);
        if (!unitDiag) {
            x[j*incX] *= A[0];
        }
        A -= ldA;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBLMV_TCC
