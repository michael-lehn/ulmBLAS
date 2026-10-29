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
ULMBLAS(chemv)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               const float        *alpha_,
               const float        *A_,
               ULMBLAS_INDEX_TYPE ldA,
               const float        *x_,
               ULMBLAS_INDEX_TYPE incX,
               const float        *beta_,
               float              *y_,
               ULMBLAS_INDEX_TYPE incY)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta  = fcomplex(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y = reinterpret_cast<fcomplex *>(y_);
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
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, false, A, 1, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, true, A, ldA, 1, x, incX,
                                       beta, y, incY);
    }
}

void
ULMBLAS(chemv_)(enum CBLAS_UPLO    upLo,
                ULMBLAS_INDEX_TYPE n,
                const float        *alpha_,
                const float        *A_,
                ULMBLAS_INDEX_TYPE ldA,
                const float        *x_,
                ULMBLAS_INDEX_TYPE incX,
                const float        *beta_,
                float              *y_,
                ULMBLAS_INDEX_TYPE incY)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta  = fcomplex(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y = reinterpret_cast<fcomplex *>(y_);
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
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, true, A, 1, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, false, A, ldA, 1, x, incX,
                                       beta, y, incY);
    }
}


void
ULMBLAS(zhemv)(enum CBLAS_UPLO    upLo,
               ULMBLAS_INDEX_TYPE n,
               const double       *alpha_,
               const double       *A_,
               ULMBLAS_INDEX_TYPE ldA,
               const double       *x_,
               ULMBLAS_INDEX_TYPE incX,
               const double       *beta_,
               double             *y_,
               ULMBLAS_INDEX_TYPE incY)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);
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
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, false, A, 1, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, true, A, ldA, 1, x, incX,
                                       beta, y, incY);
    }
}

void
ULMBLAS(zhemv_)(enum CBLAS_UPLO    upLo,
                ULMBLAS_INDEX_TYPE n,
                const double       *alpha_,
                const double       *A_,
                ULMBLAS_INDEX_TYPE ldA,
                const double       *x_,
                ULMBLAS_INDEX_TYPE incX,
                const double       *beta_,
                double             *y_,
                ULMBLAS_INDEX_TYPE incY)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);
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
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, true, A, 1, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::helmv<std::ptrdiff_t>(n, alpha, false, A, ldA, 1, x, incX,
                                       beta, y, incY);
    }
}

void
CBLAS(chemv)(enum CBLAS_ORDER   order,
             enum CBLAS_UPLO    upLo,
             ULMBLAS_INDEX_TYPE n,
             const float        *alpha,
             const float        *A,
             ULMBLAS_INDEX_TYPE ldA,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX,
             const float        *beta,
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
        CBLAS(xerbla)(info, "cblas_chemv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(chemv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(chemv_)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(zhemv)(enum CBLAS_ORDER   order,
             enum CBLAS_UPLO    upLo,
             ULMBLAS_INDEX_TYPE n,
             const double       *alpha,
             const double       *A,
             ULMBLAS_INDEX_TYPE ldA,
             const double       *x,
             ULMBLAS_INDEX_TYPE incX,
             const double       *beta,
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
        CBLAS(xerbla)(info, "cblas_zhemv", "... bla bla ...");
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(zhemv)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zhemv_)(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
