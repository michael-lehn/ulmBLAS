#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(cher2)(const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const float              *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               float                    *A_,
               const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha    = fcomplex(alpha_[0], alpha_[1]);
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
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
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("CHER2 ", &info, 6);
        return;
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
        ulmBLAS::helr2<std::ptrdiff_t>(n, false, alpha, x, incX, y, incY,
                                       A, 1, ldA);
    } else {
        ulmBLAS::helr2<std::ptrdiff_t>(n, true, alpha, x, incX, y, incY,
                                       A, ldA, 1);
    }
}


void
F77BLAS(zher2)(const char               *upLo_,
               const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               const double             *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               double                   *A_,
               const ULMBLAS_INDEX_TYPE *ldA_)
{
//
//  Dereference scalar parameters
//
    bool lower              = (toupper(*upLo_) == 'L');
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    ULMBLAS_INDEX_TYPE ldA  = *ldA_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha    = dcomplex(alpha_[0], alpha_[1]);
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
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
    } else if (incY==0) {
        info = 7;
    } else if (ldA<std::max<ULMBLAS_INDEX_TYPE>(1,n)) {
        info = 9;
    }

    if (info!=0) {
        F77BLAS(xerbla)("ZHER2 ", &info, 6);
        return;
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
        ulmBLAS::helr2<std::ptrdiff_t>(n, false, alpha, x, incX, y, incY,
                                       A, 1, ldA);
    } else {
        ulmBLAS::helr2<std::ptrdiff_t>(n, true, alpha, x, incX, y, incY,
                                       A, ldA, 1);
    }
}

} // extern "C"
