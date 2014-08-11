#ifndef ULMBLAS_SRC_LEVEL2_GEMV_TCC
#define ULMBLAS_SRC_LEVEL2_GEMV_TCC 1

#include <src/level1/scal.h>
#include <src/level1/scal.tcc>
#include <src/level2/gemv.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TX,
          typename Beta, typename TY>
void
gemv(IndexType    m,
     IndexType    n,
     const Alpha  &alpha,
     const TA     *A,
     IndexType    incRowA,
     IndexType    incColA,
     const TX     *x,
     IndexType    incX,
     const Beta   &beta,
     TY           *y,
     IndexType    incY)
{
    if (m==0 || n==0 || (alpha==Alpha(0) && beta==Beta(1))) {
        return;
    }

    scal(m, beta, y, incY);

    if (alpha==Alpha(0)) {
        return;
    }

    for (IndexType j=0; j<n; ++j) {
        for (IndexType i=0; i<m; ++i) {
            y[i*incY] += alpha*A[i*incRowA+j*incColA]*x[j*incX];
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL2_GEMV_TCC
