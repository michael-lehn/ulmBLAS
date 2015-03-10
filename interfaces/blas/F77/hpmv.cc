#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(chpmv)(const char     *upLo_,
               const int      *n_,
               const float    *alpha_,
               const float    *AP_,
               const float    *x_,
               const int      *incX_,
               const float    *beta_,
               float          *y_,
               const int      *incY_)
{
    typedef std::complex<float> fcomplex;
//
//  Dereference scalar parameters
//
    bool lowerA  = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int incX     = *incX_;
    int incY     = *incY_;

    const fcomplex *AP = reinterpret_cast<const fcomplex *>(AP_);
    const fcomplex *x  = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y  = reinterpret_cast<fcomplex *>(y_);

    fcomplex alpha     = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta      = fcomplex(beta_[0], beta_[1]);

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 6;
    } else if (incY==0) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHPMV ", &info);
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
        ulmBLAS::hplmv(n, alpha, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv(n, alpha, AP, x, incX, beta, y, incY);
    }
}


void
F77BLAS(zhpmv)(const char     *upLo_,
               const int      *n_,
               const double   *alpha_,
               const double   *AP_,
               const double   *x_,
               const int      *incX_,
               const double   *beta_,
               double         *y_,
               const int      *incY_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool lowerA  = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int incX     = *incX_;
    int incY     = *incY_;

    const dcomplex *AP = reinterpret_cast<const dcomplex *>(AP_);
    const dcomplex *x  = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y  = reinterpret_cast<dcomplex *>(y_);

    dcomplex alpha     = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta      = dcomplex(beta_[0], beta_[1]);

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 6;
    } else if (incY==0) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHPMV ", &info);
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
        ulmBLAS::hplmv(n, alpha, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv(n, alpha, AP, x, incX, beta, y, incY);
    }
}

} // extern "C"
