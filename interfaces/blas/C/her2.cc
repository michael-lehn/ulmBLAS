#include BLAS_HEADER
#include <algorithm>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(cher2)(enum CBLAS_UPLO  upLo,
               int              n,
               const float      *alpha_,
               const float      *x_,
               int              incX,
               const float      *y_,
               int              incY,
               float            *A_,
               int              ldA)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha    = fcomplex(alpha_[0], alpha_[1]);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);
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
        ulmBLAS::helr2(n, false, alpha, x, incX, y, incY, A, 1, ldA);
    } else {
        ulmBLAS::helr2(n, true, alpha, x, incX, y, incY, A, ldA, 1);
    }
}

void
ULMBLAS(cher2_)(enum CBLAS_UPLO  upLo,
                int              n,
                const float      *alpha_,
                const float      *x_,
                int              incX,
                const float      *y_,
                int              incY,
                float            *A_,
                int              ldA)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha    = fcomplex(alpha_[0], alpha_[1]);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);
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
        ulmBLAS::helr2(n, true, alpha, x, incX, y, incY, A, 1, ldA);
    } else {
        ulmBLAS::helr2(n, false, alpha, x, incX, y, incY, A, ldA, 1);
    }
}

void
ULMBLAS(zher2)(enum CBLAS_UPLO  upLo,
               int              n,
               const double     *alpha_,
               const double     *x_,
               int              incX,
               const double     *y_,
               int              incY,
               double           *A_,
               int              ldA)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha    = dcomplex(alpha_[0], alpha_[1]);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);
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
        ulmBLAS::helr2(n, false, alpha, x, incX, y, incY, A, 1, ldA);
    } else {
        ulmBLAS::helr2(n, true, alpha, x, incX, y, incY, A, ldA, 1);
    }
}

void
ULMBLAS(zher2_)(enum CBLAS_UPLO  upLo,
                int              n,
                const double     *alpha_,
                const double     *x_,
                int              incX,
                const double     *y_,
                int              incY,
                double           *A_,
                int              ldA)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha    = dcomplex(alpha_[0], alpha_[1]);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);
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
        ulmBLAS::helr2(n, true, alpha, x, incX, y, incY, A, 1, ldA);
    } else {
        ulmBLAS::helr2(n, false, alpha, x, incX, y, incY, A, ldA, 1);
    }
}

void
CBLAS(cher2)(enum CBLAS_ORDER  order,
             enum CBLAS_UPLO   upLo,
             int               n,
             const float       *alpha,
             const float       *x,
             int               incX,
             const float       *y,
             int               incY,
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
        info = (order==CblasColMajor) ? 6 : 8;
    } else if (incY==0) {
        info = (order==CblasColMajor) ? 8 : 6;
    } else if (ldA<n) {
        info = 10;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_cher2", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(cher2)(upLo, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(cher2_)(upLo, n, alpha, x, incX, y, incY, A, ldA);
    }
}

void
CBLAS(zher2)(enum CBLAS_ORDER  order,
             enum CBLAS_UPLO   upLo,
             int               n,
             const double      *alpha,
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
        info = (order==CblasColMajor) ? 6 : 8;
    } else if (incY==0) {
        info = (order==CblasColMajor) ? 8 : 6;
    } else if (ldA<n) {
        info = 10;
    }

    if (info!=0) {
        extern int RowMajorStrg;

        RowMajorStrg = (order==CblasRowMajor) ? 1 : 0;
        CBLAS(xerbla)(info, "cblas_zher2", "... bla bla ...");
    }

    if (order==CblasColMajor) {
        ULMBLAS(zher2)(upLo, n, alpha, x, incX, y, incY, A, ldA);
    } else {
        upLo = (upLo==CblasUpper) ? CblasLower : CblasUpper;
        ULMBLAS(zher2_)(upLo, n, alpha, x, incX, y, incY, A, ldA);
    }
}

} // extern "C"
