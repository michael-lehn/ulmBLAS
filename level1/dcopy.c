#include <ulmblas.h>

void
ULMBLAS(dcopy)(const int    n,
               const double *x,
               const int    incX,
               double       *y,
               const int    incY)
{
//
//  Local scalars
//
    int    i, m;

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
        m = n % 7;
        if (m!=0) {
            for (i=0; i<m; ++i) {
                y[i] = x[i];
            }
            if (n<7) {
                return;
            }
        }
        for (i=m; i<n; i+=7) {
            y[i  ] = x[i  ];
            y[i+1] = x[i+1];
            y[i+2] = x[i+2];
            y[i+3] = x[i+3];
            y[i+4] = x[i+4];
            y[i+5] = x[i+5];
            y[i+6] = x[i+6];
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
            (*y) = (*x);
        }
    }
}

void
F77BLAS(dcopy)(const int    *_n,
               const double *x,
               const int    *_incX,
               double       *y,
               const int    *_incY)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;

    ULMBLAS(dcopy)(n, x, incX, y, incY);
}
