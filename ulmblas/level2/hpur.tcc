#ifndef ULMBLAS_LEVEL2_HPUR_TCC
#define ULMBLAS_LEVEL2_HPUR_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/auxiliary/real.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/hpur.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
void
hpur(IndexType    n,
     const Alpha  &alpha,
     const TX     *x,
     IndexType    incX,
     TA           *A)
{
    if (n==0 || alpha==Alpha(0)) {
        return;
    }

    for (IndexType j=0; j<n; ++j) {
        axpy(j+1, alpha*conjugate(x[j*incX]), x, incX, A, IndexType(1));
        A[j] = real(A[j]);
        A += j+1;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPUR_TCC
