#ifndef ULMBLAS_SRC_LEVEL2_SYLMV_TCC
#define ULMBLAS_SRC_LEVEL2_SYLMV_TCC 1

#include <src/level1/scal.h>
#include <src/level1/axpy.h>
#include <src/level1extensions/dotaxpy.h>
#include <src/level2/sylmv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
sylmv(IndexType    n,
      const Alpha  &alpha,
      const TA     *A,
      IndexType    incRowA,
      IndexType    incColA,
      const TX     *x,
      IndexType    incX,
      const Beta   &beta,
      TY           *y,
      IndexType    incY)
{
    scal(n, beta, y, incY);

    TY rho;

    for (IndexType i=0; i<n; ++i) {
        dotaxpy(n-1-i, false, false, false, alpha*x[i*incX],
                &A[(i+1)*incRowA+i*incColA], incRowA,
                &x[(i+1)*incX], incX,
                &y[(i+1)*incY], incY,
                rho);
        y[i*incY] += alpha*(A[i*incRowA+i*incColA]*x[i*incX]+rho);
    }

    /*
    for (IndexType i=0; i<n; ++i) {
        axpy(i, alpha*x[i*incX], &A[i*incRowA], incColA, &y[0*incY], incY);
        y[i*incY] += alpha*A[i*incRowA+i*incColA]*x[i*incX];
        axpy(n-1-i, alpha*x[i*incX], &A[(i+1)*incRowA+i*incColA], incRowA,
             &y[(i+1)*incY], incY);
    }
    */
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL2_SYLMV_TCC
