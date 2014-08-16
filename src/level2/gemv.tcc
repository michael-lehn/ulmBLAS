#ifndef ULMBLAS_SRC_LEVEL2_GEMV_TCC
#define ULMBLAS_SRC_LEVEL2_GEMV_TCC 1

#include <src/level1/axpy.h>
#include <src/level1extensions/axpy2v.h>
#include <src/level1/scal.h>
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

    if (incRowA==IndexType(1)) {
        IndexType nb = (n/2)*2;
        IndexType nl = n % 2;

        for (IndexType j=0; j<nb; j+=2) {
            axpy2v(m, alpha*x[j*incX], alpha*x[(j+1)*incX],
                   &A[ j   *incColA], IndexType(1),
                   &A[(j+1)*incColA], IndexType(1),
                   y, incY);
        }
        if (nl) {
            axpy(m, alpha*x[(n-1)*incX],
                 &A[(n-1)*incColA], IndexType(1),
                 y, incY);
        }
     } else if (incColA==IndexType(1)) {
        TY tmp;

        for (IndexType i=0; i<m; ++i) {
            dotu(m, x, incX, &A[i*incRowA], IndexType(1), tmp);
            y[i*incY] += alpha*tmp;
        }
    } else {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<m; ++i) {
                y[i*incY] += alpha*A[i*incRowA+j*incColA]*x[j*incX];
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL2_GEMV_TCC
