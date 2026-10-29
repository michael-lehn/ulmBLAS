#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sswap)(const ULMBLAS_INDEX_TYPE n,
               float                    *x,
               const ULMBLAS_INDEX_TYPE incX,
               float                    *y,
               const ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::swap<std::ptrdiff_t>(n, x, incX, y, incY);
}

void
ULMBLAS(dswap)(const ULMBLAS_INDEX_TYPE n,
               double                   *x,
               const ULMBLAS_INDEX_TYPE incX,
               double                   *y,
               const ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::swap<std::ptrdiff_t>(n, x, incX, y, incY);
}

void
ULMBLAS(cswap)(const ULMBLAS_INDEX_TYPE n,
               float                    *x_,
               const ULMBLAS_INDEX_TYPE incX,
               float                    *y_,
               const ULMBLAS_INDEX_TYPE incY)
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
    return ulmBLAS::swap<std::ptrdiff_t>(n, x, incX, y, incY);
}

void
ULMBLAS(zswap)(const ULMBLAS_INDEX_TYPE n,
               double                   *x_,
               const ULMBLAS_INDEX_TYPE incX,
               double                   *y_,
               const ULMBLAS_INDEX_TYPE incY)
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
    return ulmBLAS::swap<std::ptrdiff_t>(n, x, incX, y, incY);
}

void
CBLAS(sswap)(const ULMBLAS_INDEX_TYPE n,
             float                    *x,
             const ULMBLAS_INDEX_TYPE incX,
             float                    *y,
             const ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(sswap)(n, x, incX, y, incY);
}

void
CBLAS(dswap)(const ULMBLAS_INDEX_TYPE n,
             double                   *x,
             const ULMBLAS_INDEX_TYPE incX,
             double                   *y,
             const ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(dswap)(n, x, incX, y, incY);
}

void
CBLAS(cswap)(const ULMBLAS_INDEX_TYPE n,
             float                    *x,
             const ULMBLAS_INDEX_TYPE incX,
             float                    *y,
             const ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(cswap)(n, x, incX, y, incY);
}

void
CBLAS(zswap)(const ULMBLAS_INDEX_TYPE n,
             double                   *x,
             const ULMBLAS_INDEX_TYPE incX,
             double                   *y,
             const ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(zswap)(n, x, incX, y, incY);
}

} // extern "C"
