#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(ssbmv)(enum CBLAS_UPLO       upLo,
               int                   n,
               int                   k,
               float                 alpha,
               const float           *A,
               int                   ldA,
               const float           *x,
               int                   incX,
               float                 beta,
               float                 *y,
               int                   incY)
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
        ulmBLAS::sblmv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::sbumv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(dsbmv)(enum CBLAS_UPLO       upLo,
               int                   n,
               int                   k,
               double                alpha,
               const double          *A,
               int                   ldA,
               const double          *x,
               int                   incX,
               double                beta,
               double                *y,
               int                   incY)
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
        ulmBLAS::sblmv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::sbumv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(ssbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             int                   n,
             int                   k,
             float                 alpha,
             const float           *A,
             int                   ldA,
             const float           *x,
             int                   incX,
             float                 beta,
             float                 *y,
             int                   incY)
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
    } else if (k<0) {
        info = 4;
    } else if (ldA<k+1) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    } else if (incY==0) {
        info = 12;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_ssbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ssbmv)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(ssbmv)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}


void
CBLAS(dsbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             int                   n,
             int                   k,
             double                alpha,
             const double          *A,
             int                   ldA,
             const double          *x,
             int                   incX,
             double                beta,
             double                *y,
             int                   incY)
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
    } else if (k<0) {
        info = 4;
    } else if (ldA<k+1) {
        info = 7;
    } else if (incX==0) {
        info = 9;
    } else if (incY==0) {
        info = 12;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dsbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsbmv)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dsbmv)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
