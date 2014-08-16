#include <interfaces/F77/config.h>
#include <src/level1/dot.h>

extern "C" {

void
F77BLAS(ddot_sub)(const int     *_n,
                  const double  *x,
                  const int     *_incX,
                  const double  *y,
                  const int     *_incY,
                  double        *_result)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    *_result = ulmBLAS::dotu(n, x, incX, y, incY);
}

} // extern "C"
