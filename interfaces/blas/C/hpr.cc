#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(chpr)(enum CBLAS_UPLO    upLo,
              ULMBLAS_INDEX_TYPE n,
              float              alpha,
              const float        *x_,
              ULMBLAS_INDEX_TYPE incX,
              float              *A_)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

    if (incX<0) {
        x -= incX*(n-1);
    }
//
//  Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::hplr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::hpur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}

void
ULMBLAS(chpr_)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               float              alpha,
               const float        *x_,
               ULMBLAS_INDEX_TYPE incX,
               float              *A_)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

    if (incX<0) {
        x -= incX*(n-1);
    }
//
//  Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::hplr<std::ptrdiff_t>(n, alpha, true, x, incX, A);
    } else {
        ulmBLAS::hpur<std::ptrdiff_t>(n, alpha, true, x, incX, A);
    }
}

void
ULMBLAS(zhpr)(enum CBLAS_UPLO    upLo,
              ULMBLAS_INDEX_TYPE n,
              double             alpha,
              const double       *x_,
              ULMBLAS_INDEX_TYPE incX,
              double             *A_)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

    if (incX<0) {
        x -= incX*(n-1);
    }
//
//  Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::hplr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::hpur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}

void
ULMBLAS(zhpr_)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               double             alpha,
               const double       *x_,
               ULMBLAS_INDEX_TYPE incX,
               double             *A_)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

    if (incX<0) {
        x -= incX*(n-1);
    }
//
//  Start the operations.
//
    if (upLo==CblasLower) {
        ulmBLAS::hplr<std::ptrdiff_t>(n, alpha, true, x, incX, A);
    } else {
        ulmBLAS::hpur<std::ptrdiff_t>(n, alpha, true, x, incX, A);
    }
}

void
CBLAS(chpr)(enum CBLAS_ORDER   order,
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
        CBLAS(xerbla)(info, "cblas_chpr", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(chpr)(upLo, n, alpha, x, incX, A);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(chpr_)(upLo, n, alpha, x, incX, A);
    }
}

void
CBLAS(zhpr)(enum CBLAS_ORDER   order,
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
        CBLAS(xerbla)(info, "cblas_zhpr", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(zhpr)(upLo, n, alpha, x, incX, A);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zhpr_)(upLo, n, alpha, x, incX, A);
    }
}

} // extern "C"
