#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sswap)(const int n,
               float     *x,
               const int incX,
               float     *y,
               const int incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::swap(n, x, incX, y, incY);
}

void
ULMBLAS(dswap)(const int n,
               double    *x,
               const int incX,
               double    *y,
               const int incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::swap(n, x, incX, y, incY);
}

void
ULMBLAS(cswap)(const int n,
               float     *x_,
               const int incX,
               float     *y_,
               const int incY)
{
    typedef std::complex<float> fcomplex;
    fcomplex *x = reinterpret_cast<fcomplex *>(x_);
    fcomplex *y = reinterpret_cast<fcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::swap(n, x, incX, y, incY);
}

void
ULMBLAS(zswap)(const int n,
               double    *x_,
               const int incX,
               double    *y_,
               const int incY)
{
    typedef std::complex<double> dcomplex;
    dcomplex *x = reinterpret_cast<dcomplex *>(x_);
    dcomplex *y = reinterpret_cast<dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::swap(n, x, incX, y, incY);
}

void
CBLAS(sswap)(const int n,
             float     *x,
             const int incX,
             float     *y,
             const int incY)
{
    ULMBLAS(sswap)(n, x, incX, y, incY);
}

void
CBLAS(dswap)(const int n,
             double    *x,
             const int incX,
             double    *y,
             const int incY)
{
    ULMBLAS(dswap)(n, x, incX, y, incY);
}

void
CBLAS(cswap)(const int n,
             float     *x,
             const int incX,
             float     *y,
             const int incY)
{
    ULMBLAS(cswap)(n, x, incX, y, incY);
}

void
CBLAS(zswap)(const int n,
             double    *x,
             const int incX,
             double    *y,
             const int incY)
{
    ULMBLAS(zswap)(n, x, incX, y, incY);
}

} // extern "C"
