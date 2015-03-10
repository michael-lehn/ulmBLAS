#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sdot_sub)(const int     *n_,
                  const float   *x,
                  const int     *incX_,
                  const float   *y,
                  const int     *incY_,
                  float         *result_)
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
    *result_ = ulmBLAS::dotu(n, x, incX, y, incY);
}

void
F77BLAS(ddot_sub)(const int     *n_,
                  const double  *x,
                  const int     *incX_,
                  const double  *y,
                  const int     *incY_,
                  double        *result_)
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
    *result_ = ulmBLAS::dotu(n, x, incX, y, incY);
}

void
F77BLAS(cdotu_sub)(const int     *n_,
                   const float   *x_,
                   const int     *incX_,
                   const float   *y_,
                   const int     *incY_,
                   double        *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex *result  = reinterpret_cast<fcomplex *>(result_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    *result = ulmBLAS::dotu(n, x, incX, y, incY);
}

void
F77BLAS(cdotc_sub)(const int     *n_,
                   const float   *x_,
                   const int     *incX_,
                   const float   *y_,
                   const int     *incY_,
                   double        *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);
    fcomplex *result  = reinterpret_cast<fcomplex *>(result_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    *result = ulmBLAS::dotc(n, x, incX, y, incY);
}

void
F77BLAS(zdotu_sub)(const int     *n_,
                   const double  *x_,
                   const int     *incX_,
                   const double  *y_,
                   const int     *incY_,
                   double        *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex *result  = reinterpret_cast<dcomplex *>(result_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    *result = ulmBLAS::dotu(n, x, incX, y, incY);
}

void
F77BLAS(zdotc_sub)(const int     *n_,
                   const double  *x_,
                   const int     *incX_,
                   const double  *y_,
                   const int     *incY_,
                   double        *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;
    int incY = *incY_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);
    dcomplex *result  = reinterpret_cast<dcomplex *>(result_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    *result = ulmBLAS::dotc(n, x, incX, y, incY);
}

} // extern "C"
