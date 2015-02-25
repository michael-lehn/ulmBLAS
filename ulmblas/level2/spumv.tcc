#ifndef ULMBLAS_LEVEL2_SPUMV_TCC
#define ULMBLAS_LEVEL2_SPUMV_TCC 1

#include <ulmblas/level1extensions/dotaxpy.h>
#include <ulmblas/level1/scal.h>
#include <ulmblas/level2/spumv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
spumv(IndexType    n,
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

        dotaxpy(j, false, false, false,
                alpha*x[j*incX],
                A, IndexType(1),
                x, incX,
                y, incY,
                rho);
        y[j*incY] += alpha*(rho+A[j]*x[j*incX]);
        A += j+1;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SPUMV_TCC
