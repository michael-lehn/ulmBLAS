#include <algorithm>
#include <cctype>
#include <cmath>
#include <interfaces/F77/config.h>
#include <interfaces/F77/xerbla.h>
#include <src/level2/gemv.h>

extern "C" {

void
F77BLAS(dgemv)(const char     *_transA,
               const int      *_m,
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
    bool transA  = (toupper(*_transA) == 'T' || toupper(*_transA) == 'C');
    int m        = *_m;
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

    if (toupper(*_transA)!='N' && toupper(*_transA)!='T'
     && toupper(*_transA)!='C' && toupper(*_transA)!='R')
    {
        info = 1;
    } else if (m<0) {
        info = 2;
    } else if (n<0) {
            info = 3;
    } else if (ldA<std::max(1,m)) {
            info = 6;
    } else if (incX==0) {
            info = 8;
    } else if (incY==0) {
            info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DGEMV ", &info);
    }

    if (!transA) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
    }

//
//  Start the operations.
//
    if (!transA) {
        ulmBLAS::gemv(m, n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::gemv(n, m, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}

} // extern "C"
