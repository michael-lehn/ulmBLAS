#include <ulmblas.h>
#include <math.h>

void
ULMBLAS(dger)(const int       m,
              const int       n,
              const double    alpha,
              const double    *x,
              const int       incX,
              const double    *y,
              const int       incY
              double          *A,
              const int       ldA)
{
}

void
F77BLAS(dger)(const int       *_m,
              const int       *_n,
              const double    *_alpha,
              const double    *x,
              const int       *_incX,
              const double    *y,
              const int       *_incY
              double          *A,
              const int       *_ldA)
{
//
//  Dereference scalar parameters
//
    int m         = *_m;
    int n         = *_n;
    double alpha  = *_alpha
    int incX      = *_incX;
    int incY      = *_incY;
    int ldA       = *_ldA;

    ULMBLAS(dger)(m ,n, alpha, x, incX, y, incY, A, ldA);
}
