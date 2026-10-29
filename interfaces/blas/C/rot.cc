#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(srot)(const ULMBLAS_INDEX_TYPE n,
              float                    *x,
              const ULMBLAS_INDEX_TYPE incX,
              float                    *y,
              const ULMBLAS_INDEX_TYPE incY,
              const float              c,
              const float              s)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
ULMBLAS(drot)(const ULMBLAS_INDEX_TYPE n,
              double                   *x,
              const ULMBLAS_INDEX_TYPE incX,
              double                   *y,
              const ULMBLAS_INDEX_TYPE incY,
              const double             c,
              const double             s)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
ULMBLAS(csrot)(const ULMBLAS_INDEX_TYPE n,
               float                    *x_,
               const ULMBLAS_INDEX_TYPE incX,
               float                    *y_,
               const ULMBLAS_INDEX_TYPE incY,
               const float              c,
               const float              s)
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
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
ULMBLAS(zdrot)(const ULMBLAS_INDEX_TYPE n,
               double                   *x_,
               const ULMBLAS_INDEX_TYPE incX,
               double                   *y_,
               const ULMBLAS_INDEX_TYPE incY,
               const double             c,
               const double             s)
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
    ulmBLAS::rot<std::ptrdiff_t>(n, x, incX, y, incY, c, s);
}

void
CBLAS(srot)(const ULMBLAS_INDEX_TYPE n,
            float                    *x,
            const ULMBLAS_INDEX_TYPE incX,
            float                    *y,
            const ULMBLAS_INDEX_TYPE incY,
            const float              c,
            const float              s)
{
    ULMBLAS(srot)(n, x, incX, y, incY, c, s);
}

void
CBLAS(drot)(const ULMBLAS_INDEX_TYPE n,
            double                   *x,
            const ULMBLAS_INDEX_TYPE incX,
            double                   *y,
            const ULMBLAS_INDEX_TYPE incY,
            const double             c,
            const double             s)
{
    ULMBLAS(drot)(n, x, incX, y, incY, c, s);
}

void
CBLAS(csrot)(const ULMBLAS_INDEX_TYPE n,
             float                    *x,
             const ULMBLAS_INDEX_TYPE incX,
             float                    *y,
             const ULMBLAS_INDEX_TYPE incY,
             const float              c,
             const float              s)
{
    ULMBLAS(csrot)(n, x, incX, y, incY, c, s);
}

void
CBLAS(zdrot)(const ULMBLAS_INDEX_TYPE n,
             double                   *x,
             const ULMBLAS_INDEX_TYPE incX,
             double                   *y,
             const ULMBLAS_INDEX_TYPE incY,
             const double             c,
             const double             s)
{
    ULMBLAS(zdrot)(n, x, incX, y, incY, c, s);
}

} // extern "C"
