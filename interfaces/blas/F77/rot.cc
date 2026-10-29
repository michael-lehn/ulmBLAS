#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(srot)(const ULMBLAS_INDEX_TYPE *n_,
              float                    *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              float                    *y,
              const ULMBLAS_INDEX_TYPE *incY_,
              const float              *c_,
              const float              *s_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    float c                 = *c_;
    float s                 = *s_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
F77BLAS(drot)(const ULMBLAS_INDEX_TYPE *n_,
              double                   *x,
              const ULMBLAS_INDEX_TYPE *incX_,
              double                   *y,
              const ULMBLAS_INDEX_TYPE *incY_,
              const double             *c_,
              const double             *s_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    double c                = *c_;
    double s                = *s_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
F77BLAS(srotg)(float *a,
               float *b,
               float *c,
               float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
F77BLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
F77BLAS(csrot)(const ULMBLAS_INDEX_TYPE *n_,
               float                    *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               float                    *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               const float              *c_,
               const float              *s_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    float c                 = *c_;
    float s                 = *s_;

    typedef std::complex<float> fcomplex;
    fcomplex *x = reinterpret_cast<fcomplex *>(x_);
    fcomplex *y = reinterpret_cast<fcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
F77BLAS(zdrot)(const ULMBLAS_INDEX_TYPE *n_,
               double                   *x_,
               const ULMBLAS_INDEX_TYPE *incX_,
               double                   *y_,
               const ULMBLAS_INDEX_TYPE *incY_,
               const double             *c_,
               const double             *s_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;
    double c                = *c_;
    double s                = *s_;

    typedef std::complex<double> dcomplex;
    dcomplex *x = reinterpret_cast<dcomplex *>(x_);
    dcomplex *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
F77BLAS(crotg)(float *a_,
               float *b_,
               float *c,
               float *s_)
{
    typedef std::complex<float> fcomplex;
    fcomplex *a = reinterpret_cast<fcomplex *>(a_);
    fcomplex *b = reinterpret_cast<fcomplex *>(b_);
    fcomplex *s = reinterpret_cast<fcomplex *>(s_);

    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
F77BLAS(zrotg)(double *a_,
               double *b_,
               double *c,
               double *s_)
{
    typedef std::complex<double> dcomplex;
    dcomplex *a = reinterpret_cast<dcomplex *>(a_);
    dcomplex *b = reinterpret_cast<dcomplex *>(b_);
    dcomplex *s = reinterpret_cast<dcomplex *>(s_);

    ulmBLAS::rotg(*a, *b, *c, *s);
}

} // extern "C"
