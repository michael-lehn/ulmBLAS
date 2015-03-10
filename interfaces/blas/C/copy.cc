#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(scopy)(int          n,
           const float  *x,
           int          incX,
           float        *y,
           int          incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::copy(n, false, x, incX, y, incY);
}

void
ULMBLAS(dcopy)(int          n,
               const double *x,
               int          incX,
               double       *y,
               int          incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::copy(n, false, x, incX, y, incY);
}

void
ULMBLAS(ccopy)(int          n,
               const float  *x_,
               int          incX,
               float        *y_,
               int          incY)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    fcomplex       *y = reinterpret_cast<fcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::copy(n, false, x, incX, y, incY);
}

void
ULMBLAS(zcopy)(int          n,
               const double *x_,
               int          incX,
               double       *y_,
               int          incY)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    dcomplex       *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::copy(n, false, x, incX, y, incY);
}

void
CBLAS(scopy)(int          n,
             const float  *x,
             int          incX,
             float        *y,
             int          incY)
{
    ULMBLAS(scopy)(n, x, incX, y, incY);
}

void
CBLAS(dcopy)(int          n,
             const double *x,
             int          incX,
             double       *y,
             int          incY)
{
    ULMBLAS(dcopy)(n, x, incX, y, incY);
}

void
CBLAS(ccopy)(int          n,
             const float  *x,
             int          incX,
             float        *y,
             int          incY)
{
    ULMBLAS(ccopy)(n, x, incX, y, incY);
}

void
CBLAS(zcopy)(int          n,
             const double *x,
             int          incX,
             double       *y,
             int          incY)
{
    ULMBLAS(zcopy)(n, x, incX, y, incY);
}

} // extern "C"
