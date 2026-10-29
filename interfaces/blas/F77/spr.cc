#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sspr)(const char               *upLo_,
              const ULMBLAS_INDEX_TYPE *n_,
              const float              *alpha_,
              const float              *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              float                    *A)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    float alpha             = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SSPR  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::splr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::spur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}

void
F77BLAS(dspr)(const char               *upLo_,
              const ULMBLAS_INDEX_TYPE *n_,
              const double             *alpha_,
              const double             *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              double                   *A)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    double alpha            = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

//
//  Test the input parameters
//
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*upLo_)!='L' && toupper(*upLo_)!='U') {
        info = 1;
    } else if (n<0) {
        info = 2;
    } else if (incX==0) {
        info = 5;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSPR  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::splr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::spur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}

} // extern "C"
