#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

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
