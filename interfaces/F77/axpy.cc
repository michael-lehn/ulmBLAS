#include <interfaces/F77/config.h>
#include <ulmblas/level1/axpy.h>

extern "C" {

void
F77BLAS(daxpy)(const int     *_n,
               const double  *_alpha,
               const double  *x,
               const int     *_incX,
               double        *y,
               int           *_incY)
{
//
//  Dereference scalar parameters
//
    int     n     = *_n;
    double  alpha = *_alpha;
    int     incX  = *_incX;
    int     incY  = *_incY;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

} // extern "C"
