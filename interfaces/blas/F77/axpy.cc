#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(saxpy)(const int     *n_,
               const float   *alpha_,
               const float   *x,
               const int     *incX_,
               float         *y,
               int           *incY_)
{
//
//  Dereference scalar parameters
//
    int     n     = *n_;
    float   alpha = *alpha_;
    int     incX  = *incX_;
    int     incY  = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

void
F77BLAS(daxpy)(const int     *n_,
               const double  *alpha_,
               const double  *x,
               const int     *incX_,
               double        *y,
               int           *incY_)
{
//
//  Dereference scalar parameters
//
    int     n     = *n_;
    double  alpha = *alpha_;
    int     incX  = *incX_;
    int     incY  = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

void
F77BLAS(caxpy)(const int     *n_,
               const float   *alpha_,
               const float   *x_,
               const int     *incX_,
               float         *y_,
               int           *incY_)
{
//
//  Dereference scalar parameters
//
    int     n     = *n_;
    int     incX  = *incX_;
    int     incY  = *incY_;

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
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

void
F77BLAS(zaxpy)(const int     *n_,
               const double  *alpha_,
               const double  *x_,
               const int     *incX_,
               double        *y_,
               int           *incY_)
{
//
//  Dereference scalar parameters
//
    int     n     = *n_;
    int     incX  = *incX_;
    int     incY  = *incY_;

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
    ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

} // extern "C"
