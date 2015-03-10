#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(saxpy)(int           n,
               float         alpha,
               const float   *x,
               int           incX,
               float         *y,
               int           incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}


void
ULMBLAS(daxpy)(int           n,
               double        alpha,
               const double  *x,
               int           incX,
               double        *y,
               int           incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

void
ULMBLAS(caxpy)(int           n,
               const float   *alpha_,
               const float   *x_,
               int           incX,
               float         *y_,
               int           incY)
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
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

void
ULMBLAS(zaxpy)(int           n,
               const double  *alpha_,
               const double  *x_,
               int           incX,
               double        *y_,
               int           incY)
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
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

void
CBLAS(saxpy)(int           n,
             float         alpha,
             const float   *x,
             int           incX,
             float         *y,
             int           incY)
{
    ULMBLAS(saxpy)(n, alpha, x, incX, y, incY);
}

void
CBLAS(daxpy)(int           n,
             double        alpha,
             const double  *x,
             int           incX,
             double        *y,
             int           incY)
{
    ULMBLAS(daxpy)(n, alpha, x, incX, y, incY);
}

void
CBLAS(caxpy)(int           n,
             const float   *alpha,
             const float   *x,
             int           incX,
             float         *y,
             int           incY)
{
    ULMBLAS(caxpy)(n, alpha, x, incX, y, incY);
}

void
CBLAS(zaxpy)(int           n,
             const double  *alpha,
             const double  *x,
             int           incX,
             double        *y,
             int           incY)
{
    ULMBLAS(zaxpy)(n, alpha, x, incX, y, incY);
}

} // extern "C"
