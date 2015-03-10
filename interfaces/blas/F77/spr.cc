#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sspr)(const char        *upLo_,
              const int         *n_,
              const float       *alpha_,
              const float       *x,
              const int         *incX_,
              float             *A)
{
//
//  Dereference scalar parameters
//
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    float alpha  = *alpha_;
    int incX     = *incX_;

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
    }

    if (info!=0) {
        F77BLAS(xerbla)("SSPR  ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::splr(n, alpha, x, incX, A);
    } else {
        ulmBLAS::spur(n, alpha, x, incX, A);
    }
}

void
F77BLAS(dspr)(const char        *upLo_,
              const int         *n_,
              const double      *alpha_,
              const double      *x,
              const int         *incX_,
              double            *A)
{
//
//  Dereference scalar parameters
//
    bool lower   = (toupper(*upLo_) == 'L');
    int n        = *n_;
    double alpha = *alpha_;
    int incX     = *incX_;

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
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSPR  ", &info);
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::splr(n, alpha, x, incX, A);
    } else {
        ulmBLAS::spur(n, alpha, x, incX, A);
    }
}

} // extern "C"
