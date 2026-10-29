#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(ssyr)(const char               *upLo_,
              const ULMBLAS_INDEX_TYPE *n_,
              const float              *alpha_,
              const float              *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              float                    *A,
              const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    float alpha             = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 7;
    }

    if (info!=0) {
        F77BLAS(xerbla)("SSYR  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::sylr<std::ptrdiff_t>(n, alpha, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::sylr<std::ptrdiff_t>(n, alpha, x, incX, A, ldA, 1);
    }
}

void
F77BLAS(dsyr)(const char               *upLo_,
              const ULMBLAS_INDEX_TYPE *n_,
              const double             *alpha_,
              const double             *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              double                   *A,
              const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    double alpha            = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

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
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 7;
    }

    if (info!=0) {
        F77BLAS(xerbla)("DSYR  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::sylr<std::ptrdiff_t>(n, alpha, x, incX, A, 1, ldA);
    } else {
        ulmBLAS::sylr<std::ptrdiff_t>(n, alpha, x, incX, A, ldA, 1);
    }
}

} // extern "C"
