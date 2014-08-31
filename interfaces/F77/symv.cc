#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <src/level2/sylmv.h>

extern "C" {

void
F77BLAS(dsymv)(const char     *_upLo,
               const int      *_n,
               const double   *_alpha,
               const double   *A,
               const int      *_ldA,
               const double   *x,
               const int      *_incX,
               const double   *_beta,
               double         *y,
               const int      *_incY)
{
//
//  Dereference scalar parameters
//
    bool lowerA  = (toupper(*_upLo) == 'L');
    int n        = *_n;
    double alpha = *_alpha;
    int ldA      = *_ldA;
    int incX     = *_incX;
    double beta  = *_beta;
    int incY     = *_incY;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*_upLo)!='U' && toupper(*_upLo)!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (ldA<std::max(1,n)) {
        info = 5;
    } else if (incX==0) {
        info = 7;
    } else if (incY==0) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSYMV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

//
//  Start the operations.
//
    if (lowerA) {
        ulmBLAS::sylmv(n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::sylmv(n, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}

} // extern "C"
