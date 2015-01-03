#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level2/splmv.h>
#include <ulmblas/level2/spumv.h>

extern "C" {

void
ULMBLAS(dspmv)(enum CBLAS_UPLO       upLo,
               int                   n,
               double                alpha,
               const double          *A,
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
        ulmBLAS::splmv(n, alpha, A, x, incX, beta, y, incY);
    } else {
        ulmBLAS::spumv(n, alpha, A, x, incX, beta, y, incY);
    }
}

void
CBLAS(dspmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             int                   n,
             double                alpha,
             const double          *A,
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
    } else if (incX==0) {
        info = 7;
    } else if (incY==0) {
        info = 10;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dspmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(dspmv)(upLo, n, alpha, A, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dspmv)(upLo, n, alpha, A, x, incX, beta, y, incY);
    }
}

} // extern "C"
