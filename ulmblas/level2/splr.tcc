#ifndef ULMBLAS_LEVEL2_SPLR_TCC
#define ULMBLAS_LEVEL2_SPLR_TCC 1

#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/splr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
void
splr(IndexType    n,
     const Alpha  &alpha,
     const TX     *x,
     IndexType    incX,
     TA           *A)
{
    for (IndexType j=0; j<n; ++j) {
        axpy(n-j, alpha*x[j*incX], &x[j*incX], incX, A, IndexType(1));
        A += n-j;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SPLR_TCC
