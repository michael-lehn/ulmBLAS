#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(chbmv)(const char     *upLo_,
               const int      *n_,
               const int      *k_,
               const float    *alpha_,
               const float    *A_,
               const int      *ldA_,
               const float    *x_,
               const int      *incX_,
               const float    *beta_,
               float          *y_,
               const int      *incY_)
{
//
//  Dereference scalar parameters
//
    bool lowerA  = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int k        = *k_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    int incY     = *incY_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta  = fcomplex(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y = reinterpret_cast<fcomplex *>(y_);


//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (k<0) {
        info = 3;
    } else if (ldA<std::max(1,k+1)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHBMV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

//
//  Start the operations.
//
    if (lowerA) {
        ulmBLAS::hblmv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hbumv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

void
F77BLAS(zhbmv)(const char     *upLo_,
               const int      *n_,
               const int      *k_,
               const double   *alpha_,
               const double   *A_,
               const int      *ldA_,
               const double   *x_,
               const int      *incX_,
               const double   *beta_,
               double         *y_,
               const int      *incY_)
{
//
//  Dereference scalar parameters
//
    bool lowerA  = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int k        = *k_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    int incY     = *incY_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta  = dcomplex(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);


//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (k<0) {
        info = 3;
    } else if (ldA<std::max(1,k+1)) {
        info = 6;
    } else if (incX==0) {
        info = 8;
    } else if (incY==0) {
        info = 11;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHBMV ", &info);
    }

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

//
//  Start the operations.
//
    if (lowerA) {
        ulmBLAS::hblmv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hbumv(n, k, alpha, A, ldA, x, incX, beta, y, incY);
    }
}

} // extern "C"
