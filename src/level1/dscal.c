#include <ulmblas.h>

void
dscal(const int    n,
      const double alpha,
      double       *x,
      const int    incX)
{
//
//  Local scalars
//
    int    i, m;

//
//  Quick return if possible
//
    if (n<=0 || incX<=0) {
        return;
    }
    if (incX==1) {
//
//      Code for increment equal to 1
//
        m = n % 5;
        if (m!=0) {
            for (i=0; i<m; ++i) {
                x[i] *= alpha;
            }
            if (n<5) {
                return;
            }
        }
        for (i=m; i<n; i+=5) {
            x[i  ] *= alpha;
            x[i+1] *= alpha;
            x[i+2] *= alpha;
            x[i+3] *= alpha;
            x[i+4] *= alpha;
        }
    } else {
//
//      Code for increment not equal to 1
//
        for (i=0; i<n; ++i, x+=incX) {
            (*x) *= alpha;
        }
    }
}

void
ULMBLAS(dscal)(const int    n,
               const double alpha,
               double       *x,
               const int    incX)
{
    dscal(n, alpha, x, incX);
}

void
F77BLAS(dscal)(const int    *_n,
               const double *_alpha,
               double       *x,
               const int    *_incX)
{
//
//  Dereference scalar parameters
//
    int n        = *_n;
    double alpha = *_alpha;
    int incX     = *_incX;

    ULMBLAS(dscal)(n, alpha, x, incX);
}
