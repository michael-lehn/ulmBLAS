#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sspr)(enum CBLAS_UPLO    upLo,
              ULMBLAS_INDEX_TYPE n,
              float              alpha,
              const float        *x,
              ULMBLAS_INDEX_TYPE incX,
              float              *A)
{
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (upLo==CblasLower) {
        ulmBLAS::splr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::spur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}

void
ULMBLAS(dspr)(enum CBLAS_UPLO    upLo,
              ULMBLAS_INDEX_TYPE n,
              double             alpha,
              const double       *x,
              ULMBLAS_INDEX_TYPE incX,
              double             *A)
{
//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (upLo==CblasLower) {
        ulmBLAS::splr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::spur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}

void
CBLAS(sspr)(enum CBLAS_ORDER   order,
            enum CBLAS_UPLO    upLo,
            ULMBLAS_INDEX_TYPE n,
            float              alpha,
            const float        *x,
            ULMBLAS_INDEX_TYPE incX,
            float              *A)
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
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_sspr", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(sspr)(upLo, n, alpha, x, incX, A);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(sspr)(upLo, n, alpha, x, incX, A);
    }

}

void
CBLAS(dspr)(enum CBLAS_ORDER   order,
            enum CBLAS_UPLO    upLo,
            ULMBLAS_INDEX_TYPE n,
            double             alpha,
            const double       *x,
            ULMBLAS_INDEX_TYPE incX,
            double             *A)
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
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_dspr", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(dspr)(upLo, n, alpha, x, incX, A);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(dspr)(upLo, n, alpha, x, incX, A);
    }

}

} // extern "C"
