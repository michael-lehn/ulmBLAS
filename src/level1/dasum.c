#include <ulmblas.h>
#include <math.h>

double
dasum(const int     n,
      const double  *x,
      const int     incX)
{
//
//  Local scalars
//
    double result = 0.0;
    int    i, m;

//
//  Quick return if possible
//
    if (n<=0 || incX<=0) {
        return result;
    }
    if (incX==1) {
//
//      Code for increment equal to 1
//
        m = n % 6;
        if (m!=0) {
            for (i=0; i<m; ++i) {
                result += fabs(x[i]);
            }
            if (n<6) {
                return result;
            }
        }
        for (i=m; i<n; i+=6) {
            result += fabs(x[i  ]);
            result += fabs(x[i+1]);
            result += fabs(x[i+2]);
            result += fabs(x[i+3]);
            result += fabs(x[i+4]);
            result += fabs(x[i+5]);
        }
    } else {
//
//      Code for increment not equal to 1
//
        if (incX<0) {
            x -= incX*(n-1);
        }
        for (i=0; i<n; ++i, x+=incX) {
            result += fabs(*x);
        }
    }

    return result;
}

double
ULMBLAS(dasum)(const int     n,
               const double  *x,
               const int     incX)
{
    return dasum(n, x, incX);
}

double
F77BLAS(dasum)(const int    *_n,
               const double *x,
               const int    *_incX)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;

    return ULMBLAS(dasum)(n, x, incX);
}


