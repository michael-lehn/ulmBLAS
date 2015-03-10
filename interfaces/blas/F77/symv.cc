#include <algorithm>
#include <cctype>
#include <cmath>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(ssymv)(const char     *upLo_,
               const int      *n_,
               const float    *alpha_,
               const float    *A,
               const int      *ldA_,
               const float    *x,
               const int      *incX_,
               const float    *beta_,
               float          *y,
               const int      *incY_)
{
//
//  Dereference scalar parameters
//
    bool lowerA  = (toupper(*upLo_) == 'L');
    int n        = *n_;
    float alpha  = *alpha_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    float beta   = *beta_;
    int incY     = *incY_;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (ldA<std::max(1,n)) {
        info = 5;
    } else if (incX==0) {
        info = 7;
    } else if (incY==0) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SSYMV ", &info);
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
        ulmBLAS::sylmv(n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::sylmv(n, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}


void
F77BLAS(dsymv)(const char     *upLo_,
               const int      *n_,
               const double   *alpha_,
               const double   *A,
               const int      *ldA_,
               const double   *x,
               const int      *incX_,
               const double   *beta_,
               double         *y,
               const int      *incY_)
{
//
//  Dereference scalar parameters
//
    bool lowerA  = (toupper(*upLo_) == 'L');
    int n        = *n_;
    double alpha = *alpha_;
    int ldA      = *ldA_;
    int incX     = *incX_;
    double beta  = *beta_;
    int incY     = *incY_;

//
//  Test the input parameters
//
    int info = 0;

    if (toupper(*upLo_)!='U' && toupper(*upLo_)!='L') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (ldA<std::max(1,n)) {
        info = 5;
    } else if (incX==0) {
        info = 7;
    } else if (incY==0) {
        info = 10;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSYMV ", &info);
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
        ulmBLAS::sylmv(n, alpha, A, 1, ldA, x, incX, beta, y, incY);
    } else {
        ulmBLAS::sylmv(n, alpha, A, ldA, 1, x, incX, beta, y, incY);
    }
}

} // extern "C"
