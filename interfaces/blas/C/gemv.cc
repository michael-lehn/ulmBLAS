#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level2/gemv.h>

//#define SCATTER

#ifdef SCATTER
#define   SCATTER_INCROWA   2
#define   SCATTER_INCCOLA   3
#define   SCATTER_INCX      4
#define   SCATTER_INCY      5
#endif


extern "C" {

void
ULMBLAS(dgemv)(enum Trans       transA,
               int              m,
               int              n,
               double           alpha,
               const double     *A,
               int              ldA,
               const double     *x,
               int              incX,
               double           beta,
               double           *y,
               int              incY)
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
        double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
        double *x_ = new double[n*incX*SCATTER_INCX];
        double *y_ = new double[m*incY*SCATTER_INCY];

        ulmBLAS::gecopy(m, n,
                        A, 1, ldA,
                        A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
        ulmBLAS::copy(n, x, incX, x_, incX*SCATTER_INCX);
        ulmBLAS::copy(m, y, incY, y_, incY*SCATTER_INCY);

//
//      Start the operations.
//
        ulmBLAS::gemv(m, n, alpha,
                      A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA,
                      x_, incX*SCATTER_INCX,
                      beta,
                      y_, incY*SCATTER_INCY);
        ulmBLAS::copy(m, y_, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
        delete [] A_;
        delete [] x_;
        delete [] y_;
    } else {
//
//      Scatter operands
//
        double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
        double *x_ = new double[m*incX*SCATTER_INCX];
        double *y_ = new double[n*incY*SCATTER_INCY];

        ulmBLAS::gecopy(m, n,
                        A, 1, ldA,
                        A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
        ulmBLAS::copy(n, x, incX, x_, incX*SCATTER_INCX);
        ulmBLAS::copy(m, y, incY, y_, incY*SCATTER_INCY);

//
//      Start the operations.
//
        ulmBLAS::gemv(n, m, alpha,
                      A_, ldA*SCATTER_INCCOLA, SCATTER_INCROWA,
                      x_, incX*SCATTER_INCX,
                      beta,
                      y_, incY*SCATTER_INCY);
        ulmBLAS::copy(m, y_, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
        delete [] A_;
        delete [] x_;
        delete [] y_;
    }
#endif
}

} // extern "C"
