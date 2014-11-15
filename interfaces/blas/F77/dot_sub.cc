#include BLAS_HEADER
#include <ulmblas/level1/dot.h>

extern "C" {

void
F77BLAS(ddot_sub)(const int     *n_,
                  const double  *x,
                  const int     *incX_,
                  const double  *y,
                  const int     *incY_,
                  double        *result_)
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
    *result_ = ulmBLAS::dotu(n, x, incX, y, incY);
}

} // extern "C"
