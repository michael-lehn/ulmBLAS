#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(saxpy)(ULMBLAS_INDEX_TYPE n,
               float              alpha,
               const float        *x,
               ULMBLAS_INDEX_TYPE incX,
               float              *y,
               ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}


void
ULMBLAS(daxpy)(ULMBLAS_INDEX_TYPE n,
               double             alpha,
               const double       *x,
               ULMBLAS_INDEX_TYPE incX,
               double             *y,
               ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}

void
ULMBLAS(caxpy)(ULMBLAS_INDEX_TYPE n,
               const float        *alpha_,
               const float        *x_,
               ULMBLAS_INDEX_TYPE incX,
               float              *y_,
               ULMBLAS_INDEX_TYPE incY)
{
    typedef std::complex<float> fcomplex;
    fcomplex       alpha = fcomplex(alpha_[0], alpha_[1]);
    const fcomplex *x    = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y    = reinterpret_cast<fcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}

void
ULMBLAS(zaxpy)(ULMBLAS_INDEX_TYPE n,
               const double       *alpha_,
               const double       *x_,
               ULMBLAS_INDEX_TYPE incX,
               double             *y_,
               ULMBLAS_INDEX_TYPE incY)
{
    typedef std::complex<double> dcomplex;
    dcomplex       alpha = dcomplex(alpha_[0], alpha_[1]);
    const dcomplex *x    = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y    = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy<std::ptrdiff_t>(n, alpha, x, incX, y, incY);
}

void
CBLAS(saxpy)(ULMBLAS_INDEX_TYPE n,
             float              alpha,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX,
             float              *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(saxpy)(n, alpha, x, incX, y, incY);
}

void
CBLAS(daxpy)(ULMBLAS_INDEX_TYPE n,
             double             alpha,
             const double       *x,
             ULMBLAS_INDEX_TYPE incX,
             double             *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(daxpy)(n, alpha, x, incX, y, incY);
}

void
CBLAS(caxpy)(ULMBLAS_INDEX_TYPE n,
             const float        *alpha,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX,
             float              *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(caxpy)(n, alpha, x, incX, y, incY);
}

void
CBLAS(zaxpy)(ULMBLAS_INDEX_TYPE n,
             const double       *alpha,
             const double       *x,
             ULMBLAS_INDEX_TYPE incX,
             double             *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(zaxpy)(n, alpha, x, incX, y, incY);
}

} // extern "C"
