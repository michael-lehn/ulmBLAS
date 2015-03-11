#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>


extern "C" {

void
ULMBLAS(dger2)(int      m,
               int      n,
               double   alpha,
               double   *x,
               int      incX,
               double   *y,
               int      incY,
               double   beta,
               double   *w,
               int      incW,
               double   *z,
               int      incZ,
               double   *A,
               int      ldA)
{
    ULMBLAS(dger)(m, n, alpha, x, incX, y, incY, A, ldA);
    ULMBLAS(dger)(m, n, beta, w, incW, z, incZ, A, ldA);
}

} // extern "C"
