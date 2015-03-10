#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sswap)(const int *n_,
               float     *x,
               const int *incX_,
               float     *y,
               const int *incY_)
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
    ulmBLAS::swap(n, x, incX, y, incY);
}

void
F77BLAS(dswap)(const int *n_,
               double    *x,
               const int *incX_,
               double    *y,
               const int *incY_)
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
    ulmBLAS::swap(n, x, incX, y, incY);
}

void
F77BLAS(cswap)(const int *n_,
               float     *x_,
               const int *incX_,
               float     *y_,
               const int *incY_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<float> fcomplex;
    fcomplex *x = reinterpret_cast<fcomplex *>(x_);
    fcomplex *y = reinterpret_cast<fcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::swap(n, x, incX, y, incY);
}

void
F77BLAS(zswap)(const int *n_,
               double    *x_,
               const int *incX_,
               double    *y_,
               const int *incY_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<double> dcomplex;
    dcomplex *x = reinterpret_cast<dcomplex *>(x_);
    dcomplex *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::swap(n, x, incX, y, incY);
}

} // extern "C"
