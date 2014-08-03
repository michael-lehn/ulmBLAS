#include <ulmblas.h>

void
dswap(const int n,
      double    *x,
      const int incX,
      double    *y,
      const int incY)
{
//
//  Local scalars
//
    int    i, m;
    double tmp;

//
//  Quick return if possible
//
    if (n==0) {
        return;
    }
    if (incX==1 && incY==1) {
//
//      Code for both increments equal to 1
//
        m = n % 3;
        if (m!=0) {
            for (i=0; i<m; ++i) {
                tmp  =  x[i];
                x[i] = y[i];
                y[i] = tmp;
            }
            if (n<3) {
                return;
            }
        }
        for (i=m; i<n; i+=3) {
            tmp  =  x[i];
            x[i] = y[i];
            y[i] = tmp;

            tmp    =  x[i+1];
            x[i+1] = y[i+1];
            y[i+1] = tmp;

            tmp    =  x[i+2];
            x[i+2] = y[i+2];
            y[i+2] = tmp;
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
            tmp = *x;
            *x  = *y;
            *y  = tmp;
        }
    }
}

void
ULMBLAS(dswap)(const int n,
               double    *x,
               const int incX,
               double    *y,
               const int incY)
{
    dswap(n, x, incX, y, incY);
}

void
F77BLAS(dswap)(const int *_n,
               double    *x,
               const int *_incX,
               double    *y,
               const int *_incY)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;

    ULMBLAS(dswap)(n, x, incX, y, incY);
}
