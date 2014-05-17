#include <ulmblas.h>
#include <stdio.h>

void
ULMBLAS(daxpy)(const int     n,
               const double  alpha,
               const double  *x,
               const int     incX,
               double        *y,
               int           incY)
{
//
//  Local scalars
//
    int    i, m;

//
//  Quick return if possible
//
    if (n<=0) {
        return;
    }
    if (alpha==0.0) {
        return;
    }
    if (incX==1 && incY==1) {
//
//      Code for both increments equal to 1
//
        m = n % 4;
        if (m!=0) {
            for (i=0; i<m; ++i) {
                y[i] += alpha*x[i];
            }
        }
        if (n<4) {
            return;
        }
        for (i=m; i<n; i+=4) {
            y[i  ] += alpha*x[i  ];
            y[i+1] += alpha*x[i+1];
            y[i+2] += alpha*x[i+2];
            y[i+3] += alpha*x[i+3];
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
            (*y) += alpha * (*x);
        }
    }
}

void
F77BLAS(daxpy)(const int     *_n,
               const double  *_alpha,
               const double  *x,
               const int     *_incX,
               double        *y,
               int           *_incY)
{
//
//  Dereference scalar parameters
//
    int     n     = *_n;
    double  alpha = *_alpha;
    int     incX  = *_incX;
    int     incY  = *_incY;

    ULMBLAS(daxpy)(n, alpha, x, incX, y, incY);
}
