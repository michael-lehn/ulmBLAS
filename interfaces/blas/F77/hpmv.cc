#include <algorithm>
#include <cctype>
#include <complex>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(chpmv)(const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *AP_,
               const float              *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const float              *beta_,
               float                    *y_,
               const ULMBLAS_INDEX_TYPE *incY_)
{
    typedef std::complex<float> fcomplex;
//
//  Dereference scalar parameters
//
    bool lowerA             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    const fcomplex *AP = reinterpret_cast<const fcomplex *>(AP_);
    const fcomplex *x  = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y  = reinterpret_cast<fcomplex *>(y_);

    fcomplex alpha     = fcomplex(alpha_[0], alpha_[1]);
    fcomplex beta      = fcomplex(beta_[0], beta_[1]);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

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
        F77BLAS(xerbla)("CHPMV ", &info, 6);
        return;
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
        ulmBLAS::hplmv<std::ptrdiff_t>(n, alpha, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv<std::ptrdiff_t>(n, alpha, AP, x, incX, beta, y, incY);
    }
}


void
F77BLAS(zhpmv)(const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *AP_,
               const double             *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const double             *beta_,
               double                   *y_,
               const ULMBLAS_INDEX_TYPE *incY_)
{
    typedef std::complex<double> dcomplex;
//
//  Dereference scalar parameters
//
    bool lowerA             = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    const dcomplex *AP = reinterpret_cast<const dcomplex *>(AP_);
    const dcomplex *x  = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y  = reinterpret_cast<dcomplex *>(y_);

    dcomplex alpha     = dcomplex(alpha_[0], alpha_[1]);
    dcomplex beta      = dcomplex(beta_[0], beta_[1]);

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

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
        F77BLAS(xerbla)("ZHPMV ", &info, 6);
        return;
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
        ulmBLAS::hplmv<std::ptrdiff_t>(n, alpha, AP, x, incX, beta, y, incY);
    } else {
        ulmBLAS::hpumv<std::ptrdiff_t>(n, alpha, AP, x, incX, beta, y, incY);
    }
}

} // extern "C"
