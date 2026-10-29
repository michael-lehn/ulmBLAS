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
ULMBLAS(ssymv)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               float              alpha,
               const float        *A,
               ULMBLAS_INDEX_TYPE ldA,
               const float        *x,
               ULMBLAS_INDEX_TYPE incX,
               float              beta,
               float              *y,
               ULMBLAS_INDEX_TYPE incY)
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
        ulmBLAS::sylmv<std::ptrdiff_t>(n, alpha, A, 1, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::sylmv<std::ptrdiff_t>(n, alpha, A, ldA, 1, x, incX,
                                       beta, y, incY);
    }
}


void
ULMBLAS(dsymv)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               double             alpha,
               const double       *A,
               ULMBLAS_INDEX_TYPE ldA,
               const double       *x,
               ULMBLAS_INDEX_TYPE incX,
               double             beta,
               double             *y,
               ULMBLAS_INDEX_TYPE incY)
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
        ulmBLAS::sylmv<std::ptrdiff_t>(n, alpha, A, 1, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::sylmv<std::ptrdiff_t>(n, alpha, A, ldA, 1, x, incX,
                                       beta, y, incY);
    }
#else
//
//  Scatter operands
//
    double *A_ = new double[ldA*n*SCATTER_INCROWA*SCATTER_INCCOLA];
    double *x_ = new double[n*incX*SCATTER_INCX];
    double *y_ = new double[n*incY*SCATTER_INCY];

    ulmBLAS::gecopy<std::ptrdiff_t>(n, n,
                                    A, 1, ldA,
                                    A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA);
    ulmBLAS::copy<std::ptrdiff_t>(n, x, incX, x_, incX*SCATTER_INCX);
    ulmBLAS::copy<std::ptrdiff_t>(n, y, incY, y_, incY*SCATTER_INCY);

//
//      Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::sylmv<std::ptrdiff_t>(n, alpha,
                                       A_, SCATTER_INCROWA, ldA*SCATTER_INCCOLA,
                                       x_, incX*SCATTER_INCX,
                                       beta,
                                       y_, incY*SCATTER_INCY);
    } else {
        ulmBLAS::sylmv<std::ptrdiff_t>(n, alpha,
                                       A_, ldA*SCATTER_INCCOLA, SCATTER_INCROWA,
                                       x_, incX*SCATTER_INCX,
                                       beta,
                                       y_, incY*SCATTER_INCY);
    }
    ulmBLAS::copy<std::ptrdiff_t>(n, y_, incY*SCATTER_INCY, y, incY);

//
//      Gather result
//
    delete [] A_;
    delete [] x_;
    delete [] y_;
#endif
}

void
CBLAS(ssymv)(enum CBLAS_ORDER   order,
             enum CBLAS_UPLO    upLo,
             ULMBLAS_INDEX_TYPE n,
             float              alpha,
             const float        *A,
             ULMBLAS_INDEX_TYPE ldA,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX,
             float              beta,
             float              *y,
             ULMBLAS_INDEX_TYPE incY)
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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
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
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(ssymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(ssymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(dsymv)(enum CBLAS_ORDER   order,
             enum CBLAS_UPLO    upLo,
             ULMBLAS_INDEX_TYPE n,
             double             alpha,
             const double       *A,
             ULMBLAS_INDEX_TYPE ldA,
             const double       *x,
             ULMBLAS_INDEX_TYPE incX,
             double             beta,
             double             *y,
             ULMBLAS_INDEX_TYPE incY)
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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
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
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dsymv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
