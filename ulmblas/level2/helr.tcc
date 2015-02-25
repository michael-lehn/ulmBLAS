#ifndef ULMBLAS_LEVEL2_HELR_TCC
#define ULMBLAS_LEVEL2_HELR_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/auxiliary/real.h>
#include <ulmblas/level2/helr.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TA>
void
helr(IndexType    n,
     const Alpha  &alpha,
     bool         conjX,
     const TX     *x,
     IndexType    incX,
     TA           *A,
     IndexType    incRowA,
     IndexType    incColA)
{
//
//  Simple reference implementation
//
    if (n==0 || alpha==Alpha(0)) {
        return;
    }

    if (!conjX) {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=j; i<n; ++i) {
                A[i*incRowA+j*incColA] += alpha*x[i*incX]*conjugate(x[j*incX]);
            }
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=j; i<n; ++i) {
                A[i*incRowA+j*incColA] += alpha*conjugate(x[i*incX])*x[j*incX];
            }
        }
    }

    for (IndexType i=0; i<n; ++i) {
        A[i*(incRowA+incColA)] = real(A[i*(incRowA+incColA)]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HELR_TCC
