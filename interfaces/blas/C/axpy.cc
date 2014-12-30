#include BLAS_HEADER
#include <ulmblas/level1/axpy.h>

extern "C" {

void
ULMBLAS(daxpy)(int           n,
               double        alpha,
               const double  *x,
               int           incX,
               double        *y,
               int           incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

void
CBLAS(daxpy)(int           n,
             double        alpha,
             const double  *x,
             int           incX,
             double        *y,
             int           incY)
{
    ULMBLAS(daxpy)(n, alpha, x, incX, y, incY);
}

} // extern "C"
