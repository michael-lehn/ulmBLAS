#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(saxpy)(const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               float                    *y,
               ULMBLAS_INDEX_TYPE       *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE     n     = *n_;
    float   alpha = *alpha_;
    ULMBLAS_INDEX_TYPE     incX  = *incX_;
    ULMBLAS_INDEX_TYPE     incY  = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}

void
F77BLAS(daxpy)(const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               double                   *y,
               ULMBLAS_INDEX_TYPE       *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE     n     = *n_;
    double  alpha = *alpha_;
    ULMBLAS_INDEX_TYPE     incX  = *incX_;
    ULMBLAS_INDEX_TYPE     incY  = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}

void
F77BLAS(caxpy)(const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               const float              *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               float                    *y_,
               ULMBLAS_INDEX_TYPE       *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    typedef std::complex<float> dcomplex;

    dcomplex  alpha(alpha_[0], alpha_[1]);

    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}

void
F77BLAS(zaxpy)(const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               const double             *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               double                   *y_,
               ULMBLAS_INDEX_TYPE       *incY_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    typedef std::complex<double> dcomplex;

    dcomplex  alpha(alpha_[0], alpha_[1]);

    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}

} // extern "C"
