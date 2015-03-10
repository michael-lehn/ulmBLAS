#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(chpr2)(const char        *upLo_,
               const int         *n_,
               const float       *alpha_,
               const float       *x_,
               const int         *incX_,
               const float       *y_,
               const int         *incY_,
               float             *A_)
{
//
//  Dereference scalar parameters
//
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int incX     = *incX_;
    int incY     = *incY_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha    = fcomplex(alpha_[0], alpha_[1]);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHPR2 ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    if (lower) {
        ulmBLAS::hplr2(n, alpha, x, incX, y, incY, A);
    } else {
        ulmBLAS::hpur2(n, alpha, x, incX, y, incY, A);
    }
}


void
F77BLAS(zhpr2)(const char        *upLo_,
               const int         *n_,
               const double      *alpha_,
               const double      *x_,
               const int         *incX_,
               const double      *y_,
               const int         *incY_,
               double            *A_)
{
//
//  Dereference scalar parameters
//
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    int incX     = *incX_;
    int incY     = *incY_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha    = dcomplex(alpha_[0], alpha_[1]);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    } else if (incY==0) {
        info = 7;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHPR2 ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    if (lower) {
        ulmBLAS::hplr2(n, alpha, x, incX, y, incY, A);
    } else {
        ulmBLAS::hpur2(n, alpha, x, incX, y, incY, A);
    }
}

} // extern "C"
