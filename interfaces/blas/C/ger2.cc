#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>


extern "C" {

void
ULMBLAS(dger2)(ULMBLAS_INDEX_TYPE m,
               ULMBLAS_INDEX_TYPE n,
               double             alpha,
               double             *x,
               ULMBLAS_INDEX_TYPE incX,
               double             *y,
               ULMBLAS_INDEX_TYPE incY,
               double             beta,
               double             *w,
               ULMBLAS_INDEX_TYPE incW,
               double             *z,
               ULMBLAS_INDEX_TYPE incZ,
               double             *A,
               ULMBLAS_INDEX_TYPE ldA)
{
    ULMBLAS(dger)(m, n, alpha, x, incX, y, incY, A, ldA);
    ULMBLAS(dger)(m, n, beta, w, incW, z, incZ, A, ldA);
}

} // extern "C"
