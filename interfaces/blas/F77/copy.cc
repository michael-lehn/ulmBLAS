#include BLAS_HEADER
#include <ulmblas/level1/copy.h>

extern "C" {

void
F77BLAS(dcopy)(const int    *n_,
               const double *x,
               const int    *incX_,
               double       *y,
               const int    *incY_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy(n, x, incX, y, incY);
}

} // extern "C"
