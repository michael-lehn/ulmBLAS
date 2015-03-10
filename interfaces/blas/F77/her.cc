#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(cher)(const char        *upLo_,
              const int         *n_,
              const float       *alpha_,
              const float       *x_,
              const int         *incX_,
              float             *A_,
              const int         *ldA_)
{
//
//  Dereference scalar parameters
//
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    float alpha  = *alpha_;
    int incX     = *incX_;
    int ldA      = *ldA_;

    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
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
    } else if (ldA<std::max(1,n)) {
        info = 7;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHER  ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::helr(n, alpha, false, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::helr(n, alpha, true, x, incX, A, ldA, 1);
    }
}

void
F77BLAS(zher)(const char        *upLo_,
              const int         *n_,
              const double      *alpha_,
              const double      *x_,
              const int         *incX_,
              double            *A_,
              const int         *ldA_)
{
//
//  Dereference scalar parameters
//
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    double alpha = *alpha_;
    int incX     = *incX_;
    int ldA      = *ldA_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
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
    } else if (ldA<std::max(1,n)) {
        info = 7;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHER  ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::helr(n, alpha, false, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::helr(n, alpha, true, x, incX, A, ldA, 1);
    }
}

} // extern "C"
