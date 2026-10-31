#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(chbmv)(enum CBLAS_UPLO       upLo,
               ULMBLAS_INDEX_TYPE    n,
               ULMBLAS_INDEX_TYPE    k,
               const float           *alpha_,
               const float           *A_,
               ULMBLAS_INDEX_TYPE    ldA,
               const float           *x_,
               ULMBLAS_INDEX_TYPE    incX,
               const float           *beta_,
               float                 *y_,
               ULMBLAS_INDEX_TYPE    incY)
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
        ulmBLAS::hblmv<std::ptrdiff_t>(n, k, alpha, A, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::hbumv<std::ptrdiff_t>(n, k, alpha, A, ldA, x, incX,
                                       beta, y, incY);
    }
}

void
ULMBLAS(chbmv_)(enum CBLAS_UPLO       upLo,
                ULMBLAS_INDEX_TYPE    n,
                ULMBLAS_INDEX_TYPE    k,
                const float           *alpha_,
                const float           *A_,
                ULMBLAS_INDEX_TYPE    ldA,
                const float           *x_,
                ULMBLAS_INDEX_TYPE    incX,
                const float           *beta_,
                float                 *y_,
                ULMBLAS_INDEX_TYPE    incY)
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
        ulmBLAS::hblmv<std::ptrdiff_t>(n, k, alpha, true, A, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::hbumv<std::ptrdiff_t>(n, k, alpha, true, A, ldA, x, incX,
                                       beta, y, incY);
    }
}


void
ULMBLAS(zhbmv)(enum CBLAS_UPLO       upLo,
               ULMBLAS_INDEX_TYPE    n,
               ULMBLAS_INDEX_TYPE    k,
               const double          *alpha_,
               const double          *A_,
               ULMBLAS_INDEX_TYPE    ldA,
               const double          *x_,
               ULMBLAS_INDEX_TYPE    incX,
               const double          *beta_,
               double                *y_,
               ULMBLAS_INDEX_TYPE    incY)
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
        ulmBLAS::hblmv<std::ptrdiff_t>(n, k, alpha, A, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::hbumv<std::ptrdiff_t>(n, k, alpha, A, ldA, x, incX,
                                       beta, y, incY);
    }
}

void
ULMBLAS(zhbmv_)(enum CBLAS_UPLO       upLo,
                ULMBLAS_INDEX_TYPE    n,
                ULMBLAS_INDEX_TYPE    k,
                const double          *alpha_,
                const double          *A_,
                ULMBLAS_INDEX_TYPE    ldA,
                const double          *x_,
                ULMBLAS_INDEX_TYPE    incX,
                const double          *beta_,
                double                *y_,
                ULMBLAS_INDEX_TYPE    incY)
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
        ulmBLAS::hblmv<std::ptrdiff_t>(n, k, alpha, true, A, ldA, x, incX,
                                       beta, y, incY);
    } else {
        ulmBLAS::hbumv<std::ptrdiff_t>(n, k, alpha, true, A, ldA, x, incX,
                                       beta, y, incY);
    }
}

void
CBLAS(chbmv)(enum CBLAS_ORDER      order,
             enum CBLAS_UPLO       upLo,
             ULMBLAS_INDEX_TYPE    n,
             ULMBLAS_INDEX_TYPE    k,
             const void            *alpha_v,
             const void            *A_v,
             ULMBLAS_INDEX_TYPE    ldA,
             const void            *x_v,
             ULMBLAS_INDEX_TYPE    incX,
             const void            *beta_v,
             void                  *y_v,
             ULMBLAS_INDEX_TYPE    incY)
{
    const float *alpha = static_cast<const float *>(alpha_v);
    const float *A = static_cast<const float *>(A_v);
    const float *x = static_cast<const float *>(x_v);
    const float *beta = static_cast<const float *>(beta_v);
    float *y = static_cast<float *>(y_v);

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
        return;
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
             ULMBLAS_INDEX_TYPE    n,
             ULMBLAS_INDEX_TYPE    k,
             const void            *alpha_v,
             const void            *A_v,
             ULMBLAS_INDEX_TYPE    ldA,
             const void            *x_v,
             ULMBLAS_INDEX_TYPE    incX,
             const void            *beta_v,
             void                  *y_v,
             ULMBLAS_INDEX_TYPE    incY)
{
    const double *alpha = static_cast<const double *>(alpha_v);
    const double *A = static_cast<const double *>(A_v);
    const double *x = static_cast<const double *>(x_v);
    const double *beta = static_cast<const double *>(beta_v);
    double *y = static_cast<double *>(y_v);

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
        return;
    }

    if (order==CblasColMajor) {
        ULMBLAS(zhbmv)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zhbmv_)(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
