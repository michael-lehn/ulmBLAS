#ifndef ULMBLAS_LEVEL2_TBLSTV_TCC
#define ULMBLAS_LEVEL2_TBLSTV_TCC 1

#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tblstv.h>

namespace ulmBLAS {

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
    if (n==0) {
        return;
    }

    A += (n-1)*ldA;
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
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBLSTV_TCC
