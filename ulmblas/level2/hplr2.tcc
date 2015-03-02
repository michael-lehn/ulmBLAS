#ifndef ULMBLAS_LEVEL2_HPLR2_TCC
#define ULMBLAS_LEVEL2_HPLR2_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/auxiliary/real.h>
#include <ulmblas/level1extensions/axpy2v.h>
#include <ulmblas/level2/hplr2.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
hplr2(IndexType    n,
      bool         conj,
      const Alpha  &alpha,
      const TX     *x,
      IndexType    incX,
      const TY     *y,
      IndexType    incY,
      TA           *A)
{
    if (n==0 || alpha==Alpha(0)) {
        return;
    }

    if (!conj) {
        for (IndexType j=0; j<n; ++j) {
            axpy2v(n-j,
                   alpha*conjugate(y[j*incY]),
                   conjugate(alpha*x[j*incX]),
                   &x[j*incX], incX,
                   &y[j*incY], incY,
                   A, IndexType(1));
            A[0] = real(A[0]);
            A += n-j;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            acxpy(n-j,
                  conjugate(alpha)*y[j*incY],
                  &x[j*incX], incX,
                  A, IndexType(1));
            acxpy(n-j,
                  alpha*x[j*incX],
                  &y[j*incY], incY,
                  A, IndexType(1));
            A[0] = real(A[0]);
            A += n-j;
        }
    }
}

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
hplr2(IndexType    n,
      const Alpha  &alpha,
      const TX     *x,
      IndexType    incX,
      const TY     *y,
      IndexType    incY,
      TA           *A)
{
    hplr2(n, false, alpha, x, incX, y, incY, A);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPLR2_TCC
