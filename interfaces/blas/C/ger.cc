#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level2/ger.h>

//#define SCATTER

#ifdef SCATTER
#define   SCATTER_INCROWA   1
#define   SCATTER_INCCOLA   1
#define   SCATTER_INCX      2
#define   SCATTER_INCY      1
#endif


extern "C" {

void
ULMBLAS(dger)(int           m,
              int           n,
              double        alpha,
              const double  *x,
              int           incX,
              const double  *y,
              int           incY,
              double        *A,
              int           ldA)
{
#ifndef SCATTER
//
//  Start the operations.
//
    ulmBLAS::ger(m, n, alpha, x, incX, y, incY, A, 1, ldA);

#else
//
//  Scatter operands
//
    double *x_ = new double[m*incX*SCATTER_INCX];
    double *y_ = new double[n*incY*SCATTER_INCY];
    double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];

    ulmBLAS::copy(m, x, incX, x_, incX*SCATTER_INCX);
    ulmBLAS::copy(m, y, incY, y_, incY*SCATTER_INCY);
    ulmBLAS::gecopy(m, n,
                    A, 1, ldA,
                    A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);

//
//  Start the operations.
//
    ulmBLAS::ger(m, n, alpha,
                 x_, incX*SCATTER_INCX,
                 y_, incY*SCATTER_INCY,
                 A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);

    ulmBLAS::gecopy(m, n,
                    A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA,
                    A, 1, ldA);
//
//  Gather result
//
    delete [] x_;
    delete [] y_;
    delete [] A_;
#endif

}

void
CBLAS(dger)(enum CBLAS_ORDER  order,
            int               m,
            int               n,
            double            alpha,
            const double      *x,
            int               incX,
            const double      *y,
            int               incY,
            double            *A,
            int               ldA)
{
//
//  Test the input parameters
//
    int ldAmin = (order==CblasColMajor) ? std::max(1,m) : std::max(1,n);
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else {
        if (order==CblasColMajor) {
            if (m<0) {
                info = 2;
            } else if (n<0) {
                info = 3;
            }
        } else {
            if (n<0) {
                info = 2;
            } else if (m<0) {
                info = 3;
            }
        }
    }
    if (info==0) {
        if (order==CblasColMajor) {
            if (incX==0) {
                info = 6;
            } else if (incY==0) {
                info = 8;
            }
        } else {
            if (incY==0) {
                info = 6;
            } else if (incX==0) {
                info = 8;
            }
        }
    }
    if (info==0) {
        if (ldA<ldAmin) {
            info = 10;
        }
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dger", "... bla bla ...");
    }
    if (incX<0) {
        x -= incX*(m-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }


    if (order==CblasColMajor) {
        ULMBLAS(dger)(m, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        ULMBLAS(dger)(n, m, alpha, y, incY, x, incX, A, ldA);
    }

}

} // extern "C"
