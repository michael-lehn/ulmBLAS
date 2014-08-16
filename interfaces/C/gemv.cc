#include <algorithm>
#include <interfaces/C/config.h>
#include <interfaces/C/xerbla.h>
#include <src/level2/gemv.h>

extern "C" {

void
ULMBLAS(dgemv)(const enum Trans  transA,
               const int         m,
               const int         n,
               const double      alpha,
               const double      *A,
               const int         ldA,
               const double      *x,
               const int         incX,
               const double      beta,
               double            *y,
               const int         incY)
{

//
//  Test the input parameters
//
    int info = 0;
    if (transA==0) {
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
        ULMBLAS(xerbla)("DGEMV ", &info);
    }

//
//  Start the operations.
//
    if (transA==NoTrans || transA==Conj) {
        if (incX<0) {
            x -= incX*(n-1);
        }
        if (incY<0) {
            y -= incY*(m-1);
        }
        ulmBLAS::gemv(m, n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        if (incX<0) {
            x -= incX*(m-1);
        }
        if (incY<0) {
            y -= incY*(n-1);
        }
        ulmBLAS::gemv(n, m, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}

} // extern "C"
