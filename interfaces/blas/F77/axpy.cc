#include BLAS_HEADER
#include <complex>
#include <ulmblas/level1/axpy.h>

extern "C" {

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
