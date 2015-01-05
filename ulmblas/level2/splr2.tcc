#ifndef ULMBLAS_LEVEL2_SPLR2_TCC
#define ULMBLAS_LEVEL2_SPLR2_TCC 1

#include <ulmblas/level1extensions/axpy2v.h>
#include <ulmblas/level2/splr2.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
splr2(IndexType    n,
      const Alpha  &alpha,
      const TX     *x,
      IndexType    incX,
      const TY     *y,
      IndexType    incY,
      TA           *A)
{
    for (IndexType j=0; j<n; ++j) {
        axpy2v(n-j,
               alpha*y[j*incY],
               alpha*x[j*incX],
               &x[j*incX], incX,
               &y[j*incY], incY,
               A, IndexType(1));
        A += n-j;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SPLR2_TCC
