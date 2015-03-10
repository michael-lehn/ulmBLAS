#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(ssyr)(enum CBLAS_UPLO  upLo,
              int              n,
              float            alpha,
              const float      *x,
              int              incX,
              float            *A,
              int              ldA)
{
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (upLo==CblasLower) {
        ulmBLAS::sylr(n, alpha, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::sylr(n, alpha, x, incX, A, ldA, 1);
    }
}

void
ULMBLAS(dsyr)(enum CBLAS_UPLO  upLo,
              int              n,
              double           alpha,
              const double     *x,
              int              incX,
              double           *A,
              int              ldA)
{
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (upLo==CblasLower) {
        ulmBLAS::sylr(n, alpha, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::sylr(n, alpha, x, incX, A, ldA, 1);
    }
}

void
CBLAS(ssyr)(enum CBLAS_ORDER  order,
            enum CBLAS_UPLO   upLo,
            int               n,
            float             alpha,
            const float       *x,
            int               incX,
            float             *A,
            int               ldA)
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
    } else if (ldA<n) {
        info = 8;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_ssyr", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(ssyr)(upLo, n, alpha, x, incX, A, ldA);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(ssyr)(upLo, n, alpha, x, incX, A, ldA);
    }

}

void
CBLAS(dsyr)(enum CBLAS_ORDER  order,
            enum CBLAS_UPLO   upLo,
            int               n,
            double            alpha,
            const double      *x,
            int               incX,
            double            *A,
            int               ldA)
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
    } else if (ldA<n) {
        info = 8;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dsyr", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsyr)(upLo, n, alpha, x, incX, A, ldA);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dsyr)(upLo, n, alpha, x, incX, A, ldA);
    }

}

} // extern "C"
