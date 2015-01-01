#ifndef ULMBLAS_LEVEL2_TBLMTV_TCC
#define ULMBLAS_LEVEL2_TBLMTV_TCC 1

#include <ulmblas/level1/dot.h>
#include <ulmblas/level2/tblmtv.h>

namespace ulmBLAS {

template <typename IndexType, typename TA, typename TX>
void
tblmtv(IndexType    n,
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

    for (IndexType j=0; j<n; ++j) {
        IndexType i0  = 0;
        IndexType i1  = std::min(1+k, n-j);
        IndexType len = std::max(IndexType(0), i1-i0);

        if (!unitDiag) {
            x[j*incX] *= A[0];
        }
        x[j*incX] += dotu(len-1, &A[1], IndexType(1), &x[(j+1)*incX], incX);
        A += ldA;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_TBLMTV_TCC
