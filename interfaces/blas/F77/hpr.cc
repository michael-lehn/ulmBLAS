#include <algorithm>
#include <cctype>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(chpr)(const char               *upLo_,
              const ULMBLAS_INDEX_TYPE *n_,
              const float              *alpha_,
              const float              *x_,
              const ULMBLAS_INDEX_TYPE *incX_,
              float                    *A_)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    float alpha             = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *A = reinterpret_cast<fcomplex *>(A_);

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
        F77BLAS(xerbla)("CHPR  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::hplr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::hpur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}


void
F77BLAS(zhpr)(const char               *upLo_,
              const ULMBLAS_INDEX_TYPE *n_,
              const double             *alpha_,
              const double             *x_,
              const ULMBLAS_INDEX_TYPE *incX_,
              double                   *A_)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    double alpha            = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *A = reinterpret_cast<dcomplex *>(A_);

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
        F77BLAS(xerbla)("ZHPR  ", &info, 6);
        return;
    }

//
//  Start the operations.
//
    if (incX<0) {
        x -= incX*(n-1);
    }

    if (lower) {
        ulmBLAS::hplr<std::ptrdiff_t>(n, alpha, x, incX, A);
    } else {
        ulmBLAS::hpur<std::ptrdiff_t>(n, alpha, x, incX, A);
    }
}

} // extern "C"
