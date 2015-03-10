#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sspr2)(enum CBLAS_UPLO  upLo,
               int              n,
               float            alpha,
               const float      *x,
               int              incX,
               const float      *y,
               int              incY,
               float            *A)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
//
//  Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::splr2(n, alpha, x, incX, y, incY, A);
    } else {
        ulmBLAS::spur2(n, alpha, x, incX, y, incY, A);
    }
}

void
ULMBLAS(dspr2)(enum CBLAS_UPLO  upLo,
               int              n,
               double           alpha,
               const double     *x,
               int              incX,
               const double     *y,
               int              incY,
               double           *A)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
//
//  Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::splr2(n, alpha, x, incX, y, incY, A);
    } else {
        ulmBLAS::spur2(n, alpha, x, incX, y, incY, A);
    }
}

void
CBLAS(sspr2)(enum CBLAS_ORDER  order,
             enum CBLAS_UPLO   upLo,
             int               n,
             float             alpha,
             const float       *x,
             int               incX,
             const float       *y,
             int               incY,
             float             *A)
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
    } else if (incX==0) {
        info = 6;
    } else if (incY==0) {
        info = 8;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_sspr2", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(sspr2)(upLo, n, alpha, x, incX, y, incY, A);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(sspr2)(upLo, n, alpha, x, incX, y, incY, A);
    }

}

void
CBLAS(dspr2)(enum CBLAS_ORDER  order,
             enum CBLAS_UPLO   upLo,
             int               n,
             double            alpha,
             const double      *x,
             int               incX,
             const double      *y,
             int               incY,
             double            *A)
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
    } else if (incX==0) {
        info = 6;
    } else if (incY==0) {
        info = 8;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dspr2", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dspr2)(upLo, n, alpha, x, incX, y, incY, A);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dspr2)(upLo, n, alpha, x, incX, y, incY, A);
    }

}

} // extern "C"
