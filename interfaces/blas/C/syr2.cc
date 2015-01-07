#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/level1/copy.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level2/sylr2.h>

extern "C" {

void
ULMBLAS(dsyr2)(enum CBLAS_UPLO  upLo,
              int              n,
              double           alpha,
              const double     *x,
              int              incX,
              const double     *y,
              int              incY,
              double           *A,
              int              ldA)
{
//
//  Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::sylr2(n, alpha, x, incX, y, incY, A, 1, ldA);
    } else {
        ulmBLAS::sylr2(n, alpha, x, incX, y, incY, A, ldA, 1);
    }
}

void
CBLAS(dsyr2)(enum CBLAS_ORDER  order,
             enum CBLAS_UPLO   upLo,
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
    } else if (ldA<n) {
        info = 10;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dsyr2", "... bla bla ...");
    }
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    if (order==CblasColMajor) {
        ULMBLAS(dsyr2)(upLo, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dsyr2)(upLo, n, alpha, x, incX, y, incY, A, ldA);
    }

}

} // extern "C"