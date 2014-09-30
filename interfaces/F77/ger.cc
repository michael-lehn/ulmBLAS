#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <ulmblas/level2/ger.h>

#include <cstdio>
#include <ulmblas/auxiliary/printmatrix.h>

extern "C" {

void
F77BLAS(dger)(const int         *_m,
              const int         *_n,
              const double      *_alpha,
              const double      *x,
              const int         *_incX,
              const double      *y,
              const int         *_incY,
              double            *A,
              const int         *_ldA)
{
//
//  Dereference scalar parameters
//
    int m        = *_m;
    int n        = *_n;
    double alpha = *_alpha;
    int incX     = *_incX;
    int incY     = *_incY;
    int ldA      = *_ldA;

//
//  Test the input parameters
//
    int info = 0;

    if (m<0) {
        info = 1;
    } else if (n<0) {
            info = 2;
    } else if (incX==0) {
            info = 5;
    } else if (incY==0) {
            info = 7;
    } else if (ldA<std::max(1,m)) {
            info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGER  ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    ulmBLAS::ger(m, n, alpha, x, incX, y, incY, A, 1, ldA);
}

} // extern "C"
