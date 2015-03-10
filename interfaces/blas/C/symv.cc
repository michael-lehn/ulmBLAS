#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

//#define SCATTER

#ifdef SCATTER
#define   SCATTER_INCROWA   2
#define   SCATTER_INCCOLA   3
#define   SCATTER_INCX      4
#define   SCATTER_INCY      5
#endif


extern "C" {

void
ULMBLAS(ssymv)(enum CBLAS_UPLO  upLo,
               int              n,
               float            alpha,
               const float      *A,
               int              ldA,
               const float      *x,
               int              incX,
               float            beta,
               float            *y,
               int              incY)
{
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    if (upLo==CblasLower) {
        ulmBLAS::sylmv(n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::sylmv(n, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}


void
ULMBLAS(dsymv)(enum CBLAS_UPLO  upLo,
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
#ifndef SCATTER
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    if (upLo==CblasLower) {
        ulmBLAS::sylmv(n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::sylmv(n, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
#else
//
//  Scatter operands
//
    double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
    double *x_ = new double[n*incX*SCATTER_INCX];
    double *y_ = new double[n*incY*SCATTER_INCY];

    ulmBLAS::gecopy(n, n,
                    A, 1, ldA,
                    A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
    ulmBLAS::copy(n, x, incX, x_, incX*SCATTER_INCX);
    ulmBLAS::copy(n, y, incY, y_, incY*SCATTER_INCY);

//
//      Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::sylmv(n, alpha,
                       A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA,
                       x_, incX*SCATTER_INCX,
                       beta,
                       y_, incY*SCATTER_INCY);
    } else {
        ulmBLAS::sylmv(n, alpha,
                       A_, ldA*SCATTER_INCCOLA, SCATTER_INCROWA,
                       x_, incX*SCATTER_INCX,
                       beta,
                       y_, incY*SCATTER_INCY);
    }
    ulmBLAS::copy(n, y_, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
    delete [] A_;
    delete [] x_;
    delete [] y_;
#endif
}

void
CBLAS(ssymv)(enum CBLAS_ORDER  order,
             enum CBLAS_UPLO   upLo,
             int               n,
             float             alpha,
             const float       *A,
             int               ldA,
             const float       *x,
             int               incX,
             float             beta,
             float             *y,
             int               incY)
{
//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasUpper && upLo!=CblasLower) {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_ssymv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ssymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(ssymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(dsymv)(enum CBLAS_ORDER  order,
             enum CBLAS_UPLO   upLo,
             int               n,
             double            alpha,
             const double      *A,
             int               ldA,
             const double      *x,
             int               incX,
             double            beta,
             double            *y,
             int               incY)
{
//
//  Test the input parameters
//
    int info = 0;
    if (order!=CblasColMajor && order!=CblasRowMajor) {
        info = 1;
    } else if (upLo!=CblasUpper && upLo!=CblasLower) {
        info = 2;
    } else if (n<0) {
        info = 3;
    } else if (ldA<std::max(1,n)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dsymv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dsymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
