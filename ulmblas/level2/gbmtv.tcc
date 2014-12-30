#ifndef ULMBLAS_LEVEL2_GBMTV_TCC
#define ULMBLAS_LEVEL2_GBMTV_TCC 1

#include <ulmblas/level1/dot.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level2/gbmtv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gbmtv(IndexType    m,
      IndexType    n,
      IndexType    kl,
      IndexType    ku,
      const Alpha  &alpha,
      const TA     *A,
      IndexType    ldA,
      const TX     *x,
      IndexType    incX,
      const Beta   &beta,
      TY           *y,
      IndexType    incY)
{
    if (m==0 || n==0 || (alpha==Alpha(0) && beta==Beta(1))) {
        return;
    }

    scal(n, beta, y, incY);

    if (alpha==Alpha(0)) {
        return;
    }

    for (IndexType j=0; j<n; ++j) {
        IndexType i0  = std::max(IndexType(0), ku-j);
        IndexType i1  = std::min(ku+1+kl, m+ku-j);
        IndexType len = std::max(IndexType(0), i1-i0);

        IndexType iX  = std::max(IndexType(0), j-ku);

        y[j*incY] += alpha*dotu(len, &A[i0], IndexType(1), &x[iX*incX], incX);
        A += ldA;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_GBMTV_TCC
