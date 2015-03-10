#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(srot)(const int      *n_,
              float          *x,
              const int      *incX_,
              float          *y,
              const int      *incY_,
              const float    *c_,
              const float    *s_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;
    float c  = *c_;
    float s  = *s_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot(n, x, incX, y, incY, c, s);
}

void
F77BLAS(drot)(const int      *n_,
              double         *x,
              const int      *incX_,
              double         *y,
              const int      *incY_,
              const double   *c_,
              const double   *s_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;
    double c = *c_;
    double s = *s_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot(n, x, incX, y, incY, c, s);
}

void
F77BLAS(srotg)(float *a,
               float *b,
               float *c,
               float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
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
