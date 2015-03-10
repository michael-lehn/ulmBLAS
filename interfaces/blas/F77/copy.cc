#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(scopy)(const int    *n_,
               const float  *x,
               const int    *incX_,
               float        *y,
               const int    *incY_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy(n, false, x, incX, y, incY);
}

void
F77BLAS(dcopy)(const int    *n_,
               const double *x,
               const int    *incX_,
               double       *y,
               const int    *incY_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy(n, false, x, incX, y, incY);
}

void
F77BLAS(ccopy)(const int    *n_,
               const float  *x_,
               const int    *incX_,
               float        *y_,
               const int    *incY_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<float> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy(n, false, x, incX, y, incY);
}

void
F77BLAS(zcopy)(const int    *n_,
               const double *x_,
               const int    *incX_,
               double       *y_,
               const int    *incY_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::copy(n, false, x, incX, y, incY);
}

} // extern "C"
