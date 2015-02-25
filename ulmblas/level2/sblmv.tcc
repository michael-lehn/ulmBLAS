#ifndef ULMBLAS_LEVEL2_SBLMV_TCC
#define ULMBLAS_LEVEL2_SBLMV_TCC 1

#include <ulmblas/level1extensions/dotaxpy.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level2/sblmv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
sblmv(IndexType    n,
      IndexType    k,
      const Alpha  &alpha,
      const TA     *A,
      IndexType    ldA,
      const TX     *x,
      IndexType    incX,
      const Beta   &beta,
      TY           *y,
      IndexType    incY)
{
    typedef decltype(Alpha(0)*TA(0)*TX(0)+Beta(0)*TY(0))  T;

    if (n==0 || (alpha==Alpha(0) && beta==Beta(1))) {
        return;
    }

    scal(n, beta, y, incY);

    if (alpha==Alpha(0)) {
        return;
    }

    for (IndexType j=0; j<n; ++j) {
        IndexType i1  = k - std::max(IndexType(0), (j+1+k)-n);
        IndexType len = std::max(IndexType(0), i1+1);
        T         rho;

        dotaxpy(len-1, false, false, false,
                alpha*x[j*incX],
                &A[1], IndexType(1),
                &x[(j+1)*incX], incX,
                &y[(j+1)*incY], incY,
                rho);
        y[j*incY] += alpha*(rho+A[0]*x[j*incX]);
        A += ldA;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SBLMV_TCC
