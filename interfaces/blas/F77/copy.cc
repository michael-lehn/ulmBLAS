#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(scopy)(const ULMBLAS_INDEX_TYPE *n_,
               const float              *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               float                    *y,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

void
F77BLAS(dcopy)(const ULMBLAS_INDEX_TYPE *n_,
               const double             *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               double                   *y,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

void
F77BLAS(ccopy)(const ULMBLAS_INDEX_TYPE *n_,
               const float              *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               float                    *y_,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    typedef std::complex<float> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

void
F77BLAS(zcopy)(const ULMBLAS_INDEX_TYPE *n_,
               const double             *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               double                   *y_,
               const ULMBLAS_INDEX_TYPE *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

} // extern "C"
