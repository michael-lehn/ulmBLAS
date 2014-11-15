#include BLAS_HEADER
#include <ulmblas/level1/axpy.h>

extern "C" {

void
F77BLAS(daxpy)(const int     *n_,
               const double  *alpha_,
               const double  *x,
               const int     *incX_,
               double        *y,
               int           *incY_)
{
//
//  Dereference scalar parameters
//
    int     n     = *n_;
    double  alpha = *alpha_;
    int     incX  = *incX_;
    int     incY  = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

} // extern "C"
