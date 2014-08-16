#include <interfaces/F77/config.h>
#include <src/level1/rot.h>

extern "C" {

void
F77BLAS(drot)(const int      *_n,
              double         *x,
              const int      *_incX,
              double         *y,
              const int      *_incY,
              const double   *_c,
              const double   *_s)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;
    double c = *_c;
    double s = *_s;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot(n, x, incX, y, incY, c, s);
}

void
F77BLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}
} // extern "C"
