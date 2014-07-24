#include <ulmblas.h>
#include <auxiliary/xerbla.h>
#include <math.h>

void
dger(const int     m,
     const int     n,
     const double  alpha,
     const double  *x,
     const int     incX,
     const double  *y,
     const int     incY,
     double        *A,
     const int     incRowA,
     const int     incColA)
{
    int i, j;

    if (incX<0) {
        x -= (m-1)*incX;
    }
    if (incY<0) {
        y -= (n-1)*incY;
    }

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            A[i*incRowA+j*incColA] += alpha*x[i*incX]*y[j*incY];
        }
    }
}

void
ULMBLAS(dger)(const int     m,
              const int     n,
              const double  alpha,
              const double  *x,
              const int     incX,
              const double  *y,
              const int     incY,
              double        *A,
              const int     ldA)
{
    dger(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

void
F77BLAS(dger)(const int     *_m,
              const int     *_n,
              const double  *_alpha,
              const double  *x,
              const int     *_incX,
              const double  *y,
              const int     *_incY,
              double        *A,
              const int     *_ldA)
{
    int m        = *_m;
    int n        = *_n;
    double alpha = *_alpha;
    int incX     = *_incX;
    int incY     = *_incY;
    int ldA      = *_ldA;

    int info = 0;
    if (m<0) {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    } else if (ldA<max(1,m)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGER  ", &info);
    }

    return ULMBLAS(dger)(m, n, alpha, x, incX, y, incY, A, ldA);
}


