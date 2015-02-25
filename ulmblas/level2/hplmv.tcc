#ifndef ULMBLAS_LEVEL2_HPLMV_TCC
#define ULMBLAS_LEVEL2_HPLMV_TCC 1

#include <ulmblas/auxiliary/real.h>
#include <ulmblas/level1extensions/dotaxpy.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level2/hplmv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
hplmv(IndexType    n,
      const Alpha  &alpha,
      const TA     *A,
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
        T  rho;

        dotaxpy(n-1-j, false, true, false,
                alpha*x[j*incX],
                &A[1], IndexType(1),
                &x[(j+1)*incX], incX,
                &y[(j+1)*incY], incY,
                rho);
        y[j*incY] += alpha*(rho+real(A[0])*x[j*incX]);
        A += n-j;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPLMV_TCC
