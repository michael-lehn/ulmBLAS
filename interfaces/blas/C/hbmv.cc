#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(chbmv)(enum CBLAS_UPLO       upLo,
               int                   n,
               int                   k,
               const float           *alpha_,
               const float           *A_,
               int                   ldA,
               const float           *x_,
               int                   incX,
               const float           *beta_,
               float                 *y_,
               int                   incY)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta  = fcomplex(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y = reinterpret_cast<fcomplex *>(y_);

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
        ulmBLAS::hblmv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hbumv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(chbmv_)(enum CBLAS_UPLO       upLo,
                int                   n,
                int                   k,
                const float           *alpha_,
                const float           *A_,
                int                   ldA,
                const float           *x_,
                int                   incX,
                const float           *beta_,
                float                 *y_,
                int                   incY)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta  = fcomplex(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y = reinterpret_cast<fcomplex *>(y_);

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
        ulmBLAS::hblmv(n, k, alpha, true, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hbumv(n, k, alpha, true, A, ldA, x, incX, beta, y, incY);
    }
}


void
ULMBLAS(zhbmv)(enum CBLAS_UPLO       upLo,
               int                   n,
               int                   k,
               const double          *alpha_,
               const double          *A_,
               int                   ldA,
               const double          *x_,
               int                   incX,
               const double          *beta_,
               double                *y_,
               int                   incY)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

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
        ulmBLAS::hblmv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hbumv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(zhbmv_)(enum CBLAS_UPLO       upLo,
                int                   n,
                int                   k,
                const double          *alpha_,
                const double          *A_,
                int                   ldA,
                const double          *x_,
                int                   incX,
                const double          *beta_,
                double                *y_,
                int                   incY)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

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
        ulmBLAS::hblmv(n, k, alpha, true, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hbumv(n, k, alpha, true, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(chbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             int                   n,
             int                   k,
             const float           *alpha,
             const float           *A,
             int                   ldA,
             const float           *x,
             int                   incX,
             const float           *beta,
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
        CBLAS(xerbla)(info, "cblas_chbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(chbmv)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(chbmv_)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
CBLAS(zhbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             int                   n,
             int                   k,
             const double          *alpha,
             const double          *A,
             int                   ldA,
             const double          *x,
             int                   incX,
             const double          *beta,
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
        CBLAS(xerbla)(info, "cblas_zhbmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zhbmv)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zhbmv_)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
