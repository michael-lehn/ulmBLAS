#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(chpmv)(enum CBLAS_UPLO       upLo,
               int                   n,
               const float           *alpha_,
               const float           *AP_,
               const float           *x_,
               int                   incX,
               const float           *beta_,
               float                 *y_,
               int                   incY)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *AP = reinterpret_cast<const fcomplex *>(AP_);
    const fcomplex *x  = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y  = reinterpret_cast<fcomplex *>(y_);

    fcomplex alpha     = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta      = fcomplex(beta_[0], beta_[1]);

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
        ulmBLAS::hplmv(n, alpha, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv(n, alpha, AP, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(chpmv_)(enum CBLAS_UPLO       upLo,
                int                   n,
                const float           *alpha_,
                const float           *AP_,
                const float           *x_,
                int                   incX,
                const float           *beta_,
                float                 *y_,
                int                   incY)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *AP = reinterpret_cast<const fcomplex *>(AP_);
    const fcomplex *x  = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y  = reinterpret_cast<fcomplex *>(y_);

    fcomplex alpha     = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta      = fcomplex(beta_[0], beta_[1]);

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
        ulmBLAS::hplmv(n, alpha, true, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv(n, alpha, true, AP, x, incX, beta, y, incY);
    }
}


void
ULMBLAS(zhpmv)(enum CBLAS_UPLO       upLo,
               int                   n,
               const double          *alpha_,
               const double          *AP_,
               const double          *x_,
               int                   incX,
               const double          *beta_,
               double                *y_,
               int                   incY)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *AP = reinterpret_cast<const dcomplex *>(AP_);
    const dcomplex *x  = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y  = reinterpret_cast<dcomplex *>(y_);

    dcomplex alpha     = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta      = dcomplex(beta_[0], beta_[1]);

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
        ulmBLAS::hplmv(n, alpha, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv(n, alpha, AP, x, incX, beta, y, incY);
    }
}

void
ULMBLAS(zhpmv_)(enum CBLAS_UPLO       upLo,
                int                   n,
                const double          *alpha_,
                const double          *AP_,
                const double          *x_,
                int                   incX,
                const double          *beta_,
                double                *y_,
                int                   incY)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *AP = reinterpret_cast<const dcomplex *>(AP_);
    const dcomplex *x  = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y  = reinterpret_cast<dcomplex *>(y_);

    dcomplex alpha     = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta      = dcomplex(beta_[0], beta_[1]);

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
        ulmBLAS::hplmv(n, alpha, true, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv(n, alpha, true, AP, x, incX, beta, y, incY);
    }
}

void
CBLAS(chpmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             int                   n,
             const float           *alpha,
             const float           *AP,
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
    } else if (incX==0) {
        info = 7;
    } else if (incY==0) {
        info = 10;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_chpmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(chpmv)(upLo, n, alpha, AP, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(chpmv_)(upLo, n, alpha, AP, x, incX, beta, y, incY);
    }
}

void
CBLAS(zhpmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             int                   n,
             const double          *alpha,
             const double          *AP,
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
    } else if (incX==0) {
        info = 7;
    } else if (incY==0) {
        info = 10;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_zhpmv", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zhpmv)(upLo, n, alpha, AP, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zhpmv_)(upLo, n, alpha, AP, x, incX, beta, y, incY);
    }
}

} // extern "C"
