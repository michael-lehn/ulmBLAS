#include <ulmblas.h>

double
ddot(const int     n,
     const double  *x,
     const int     incX,
     const double  *y,
     const int     incY)
{
//
//  Local scalars
//
    double result = 0.0;

    int    i, m;

//
//  Quick return if possible
//
    if (n==0) {
        return 0;
    }
    if (incX==1 && incY==1) {
//
//      Code for both increments equal to 1
//
        m = n % 5;
        if (m!=0) {
            for (i=0; i<m; ++i) {
                result += x[i] * y[i];
            }
            if (n<5) {
                return result;
            }
        }
        for (i=m; i<n; i+=5) {
            result += x[i  ] * y[i  ] ;
            result += x[i+1] * y[i+1];
            result += x[i+2] * y[i+2];
            result += x[i+3] * y[i+3];
            result += x[i+4] * y[i+4];
        }
    } else {
//
//      Code for unequal increments or equal increments not equal to 1
//
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
        for (i=0; i<n; ++i, x+=incX, y+=incY) {
            result += (*x) * (*y);
        }
    }
    return result;
}

double
ULMBLAS(ddot)(const int     n,
              const double  *x,
              const int     incX,
              const double  *y,
              const int     incY)
{
    return ddot(n, x, incX, y, incY);
}

double
F77BLAS(ddot)(const int     *_n,
              const double  *x,
              const int     *_incX,
              const double  *y,
              const int     *_incY)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;

    return ULMBLAS(ddot)(n, x, incX, y, incY);
}


