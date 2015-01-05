#ifndef ULMBLAS_LEVEL2_SPUR_TCC
#define ULMBLAS_LEVEL2_SPUR_TCC 1

#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/spur.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
void
spur(IndexType    n,
     const Alpha  &alpha,
     const TX     *x,
     IndexType    incX,
     TA           *A)
{
    for (IndexType j=0; j<n; ++j) {
        axpy(j+1, alpha*x[j*incX], x, incX, A, IndexType(1));
        A += j+1;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_SPUR_TCC
