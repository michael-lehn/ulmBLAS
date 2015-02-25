#ifndef ULMBLAS_LEVEL2_HPUR2_TCC
#define ULMBLAS_LEVEL2_HPUR2_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/auxiliary/real.h>
#include <ulmblas/level1extensions/axpy2v.h>
#include <ulmblas/level2/hpur2.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
hpur2(IndexType    n,
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

    for (IndexType j=0; j<n; ++j) {
        axpy2v(j+1,
               alpha*conjugate(y[j*incY]),
               conjugate(alpha*x[j*incX]),
               x, incX,
               y, incY,
               A, IndexType(1));
        A[j] = real(A[j]);
        A += j+1;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPUR2_TCC
