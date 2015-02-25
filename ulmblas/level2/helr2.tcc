#ifndef ULMBLAS_LEVEL2_HELR2_TCC
#define ULMBLAS_LEVEL2_HELR2_TCC 1

#include <ulmblas/level2/helr2.h>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/auxiliary/real.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TX, typename TY,
          typename TA>
void
helr2(IndexType    n,
      bool         conj,
      const Alpha  &alpha,
      const TX     *x,
      IndexType    incX,
      const TY     *y,
      IndexType    incY,
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

    TA z;

    for (IndexType j=0; j<n; ++j) {
        for (IndexType i=j; i<n; ++i) {
            z  = alpha            * x[i*incX] *conjugate(y[j*incY]);
            z += conjugate(alpha) * y[i*incY] *conjugate(x[j*incX]);
            A[i*incRowA+j*incColA] += conjugate(z, conj);
        }
    }
    for (IndexType j=0; j<n; ++j) {
        A[j*(incRowA+incColA)] = real(A[j*(incRowA+incColA)]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL2_HELR2_TCC
