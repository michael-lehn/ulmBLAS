#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(scopy)(ULMBLAS_INDEX_TYPE n,
           const float            *x,
           ULMBLAS_INDEX_TYPE     incX,
           float                  *y,
           ULMBLAS_INDEX_TYPE     incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

void
ULMBLAS(dcopy)(ULMBLAS_INDEX_TYPE n,
               const double       *x,
               ULMBLAS_INDEX_TYPE incX,
               double             *y,
               ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

void
ULMBLAS(ccopy)(ULMBLAS_INDEX_TYPE n,
               const float        *x_,
               ULMBLAS_INDEX_TYPE incX,
               float              *y_,
               ULMBLAS_INDEX_TYPE incY)
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
    return ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

void
ULMBLAS(zcopy)(ULMBLAS_INDEX_TYPE n,
               const double       *x_,
               ULMBLAS_INDEX_TYPE incX,
               double             *y_,
               ULMBLAS_INDEX_TYPE incY)
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
    return ulmBLAS::copy<std::ptrdiff_t>(n, false, x, incX, y, incY);
}

void
CBLAS(scopy)(ULMBLAS_INDEX_TYPE n,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX,
             float              *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(scopy)(n, x, incX, y, incY);
}

void
CBLAS(dcopy)(ULMBLAS_INDEX_TYPE n,
             const double       *x,
             ULMBLAS_INDEX_TYPE incX,
             double             *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(dcopy)(n, x, incX, y, incY);
}

void
CBLAS(ccopy)(ULMBLAS_INDEX_TYPE n,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX,
             float              *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(ccopy)(n, x, incX, y, incY);
}

void
CBLAS(zcopy)(ULMBLAS_INDEX_TYPE n,
             const double       *x,
             ULMBLAS_INDEX_TYPE incX,
             double             *y,
             ULMBLAS_INDEX_TYPE incY)
{
    ULMBLAS(zcopy)(n, x, incX, y, incY);
}

} // extern "C"
