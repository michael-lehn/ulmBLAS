#ifndef ULMBLAS_LEVEL2_SPUR2_TCC
#define ULMBLAS_LEVEL2_SPUR2_TCC 1

#include <ulmblas/level1extensions/axpy2v.h>
#include <ulmblas/level2/spur2.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
spur2(IndexType    n,
      const Alpha  &alpha,
      const TX     *x,
      IndexType    incX,
      const TY     *y,
      IndexType    incY,
      TA           *A)
{
    for (IndexType j=0; j<n; ++j) {
        axpy2v(j+1,
               alpha*y[j*incY],
               alpha*x[j*incX],
               x, incX,
               y, incY,
               A, IndexType(1));
        A += j+1;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SPUR2_TCC
