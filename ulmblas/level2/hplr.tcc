#ifndef ULMBLAS_LEVEL2_HPLR_TCC
#define ULMBLAS_LEVEL2_HPLR_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/auxiliary/real.h>
#include <ulmblas/level1/axpy.h>
#include <ulmblas/level2/hplr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
void
hplr(IndexType    n,
     const Alpha  &alpha,
     bool         conjX,
     const TX     *x,
     IndexType    incX,
     TA           *A)
{
    if (n==0 || alpha==Alpha(0)) {
        return;
    }

    if (!conjX) {
        for (IndexType j=0; j<n; ++j) {
            axpy(n-j, alpha*conjugate(x[j*incX]),
                 &x[j*incX], incX,
                 A, IndexType(1));
            A[0] = real(A[0]);
            A += n-j;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            acxpy(n-j, alpha*x[j*incX], &x[j*incX], incX, A, IndexType(1));
            A[0] = real(A[0]);
            A += n-j;
        }
    }
}

template <typename IndexType, typename Alpha, typename TX, typename TA>
void
hplr(IndexType    n,
     const Alpha  &alpha,
     const TX     *x,
     IndexType    incX,
     TA           *A)
{
    hplr(n, alpha, false, x, incX, A);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HPLR_TCC
