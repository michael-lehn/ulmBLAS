#include <algorithm>
#include <interfaces/C/config.h>
#include <interfaces/C/xerbla.h>
#include <src/level1/copy.h>
#include <src/level1extensions/gecopy.h>
#include <src/level2/gemv.h>

//#define SCATTER

#ifdef SCATTER
#define   SCATTER_INCROWA   2
#define   SCATTER_INCCOLA   3
#define   SCATTER_INCX      4
#define   SCATTER_INCY      5
#endif


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
    if (transA!=NoTrans && transA!=Conj
     && transA!=Trans && transA!=ConjTrans)
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
        ULMBLAS(xerbla)("DGEMV ", &info);
    }

#ifndef SCATTER
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
#else
    if (transA==NoTrans || transA==Conj) {
//
//      Scatter operands
//
        double *_A = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
        double *_x = new double[n*incX*SCATTER_INCX];
        double *_y = new double[m*incY*SCATTER_INCY];

        ulmBLAS::gecopy(m, n,
                        A, 1, ldA,
                        _A, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
        ulmBLAS::copy(n, x, incX, _x, incX*SCATTER_INCX);
        ulmBLAS::copy(m, y, incY, _y, incY*SCATTER_INCY);

//
//      Start the operations.
//
        ulmBLAS::gemv(m, n, alpha,
                      _A, SCATTER_INCROWA, ldA*SCATTER_INCCOLA,
                      _x, incX*SCATTER_INCX,
                      beta,
                      _y, incY*SCATTER_INCY);
        ulmBLAS::copy(m, _y, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
        delete [] _A;
        delete [] _x;
        delete [] _y;
    } else {
//
//      Scatter operands
//
        double *_A = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
        double *_x = new double[m*incX*SCATTER_INCX];
        double *_y = new double[n*incY*SCATTER_INCY];

        ulmBLAS::gecopy(m, n,
                        A, 1, ldA,
                        _A, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
        ulmBLAS::copy(n, x, incX, _x, incX*SCATTER_INCX);
        ulmBLAS::copy(m, y, incY, _y, incY*SCATTER_INCY);

//
//      Start the operations.
//
        ulmBLAS::gemv(n, m, alpha,
                      _A, ldA*SCATTER_INCCOLA, SCATTER_INCROWA,
                      _x, incX*SCATTER_INCX,
                      beta,
                      _y, incY*SCATTER_INCY);
        ulmBLAS::copy(m, _y, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
        delete [] _A;
        delete [] _x;
        delete [] _y;
    }
#endif
}

} // extern "C"
