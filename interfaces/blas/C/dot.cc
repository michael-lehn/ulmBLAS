#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

float
ULMBLAS(sdot)(ULMBLAS_INDEX_TYPE n,
              const float        *x,
              ULMBLAS_INDEX_TYPE incX,
              const float        *y,
              ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::dotu<std::ptrdiff_t>(n, x, incX, y, incY);
}

double
ULMBLAS(ddot)(ULMBLAS_INDEX_TYPE n,
              const double       *x,
              ULMBLAS_INDEX_TYPE incX,
              const double       *y,
              ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::dotu<std::ptrdiff_t>(n, x, incX, y, incY);
}

void
ULMBLAS(cdotc_sub)(ULMBLAS_INDEX_TYPE n,
                   const float        *x_,
                   ULMBLAS_INDEX_TYPE incX,
                   const float        *y_,
                   ULMBLAS_INDEX_TYPE incY,
                   float              *dotc_)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    fcomplex dotc = ulmBLAS::dotc<std::ptrdiff_t>(n, x, incX, y, incY);
    dotc_[0] = std::real(dotc);
    dotc_[1] = std::imag(dotc);
}

void
ULMBLAS(zdotc_sub)(ULMBLAS_INDEX_TYPE n,
                   const double       *x_,
                   ULMBLAS_INDEX_TYPE incX,
                   const double       *y_,
                   ULMBLAS_INDEX_TYPE incY,
                   double             *dotc_)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    dcomplex dotc = ulmBLAS::dotc<std::ptrdiff_t>(n, x, incX, y, incY);
    dotc_[0] = std::real(dotc);
    dotc_[1] = std::imag(dotc);
}

void
ULMBLAS(cdotu_sub)(ULMBLAS_INDEX_TYPE n,
                   const float        *x_,
                   ULMBLAS_INDEX_TYPE incX,
                   const float        *y_,
                   ULMBLAS_INDEX_TYPE incY,
                   float              *dotu_)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);
    const fcomplex *y = reinterpret_cast<const fcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    fcomplex dotu = ulmBLAS::dotu<std::ptrdiff_t>(n, x, incX, y, incY);
    dotu_[0] = std::real(dotu);
    dotu_[1] = std::imag(dotu);
}

void
ULMBLAS(zdotu_sub)(ULMBLAS_INDEX_TYPE n,
                   const double       *x_,
                   ULMBLAS_INDEX_TYPE incX,
                   const double       *y_,
                   ULMBLAS_INDEX_TYPE incY,
                   double             *dotu_)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);
    const dcomplex *y = reinterpret_cast<const dcomplex *>(y_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }

    dcomplex dotu = ulmBLAS::dotu<std::ptrdiff_t>(n, x, incX, y, incY);
    dotu_[0] = std::real(dotu);
    dotu_[1] = std::imag(dotu);
}

float
CBLAS(sdot)(ULMBLAS_INDEX_TYPE n,
            const float        *x,
            ULMBLAS_INDEX_TYPE incX,
            const float        *y,
            ULMBLAS_INDEX_TYPE incY)
{
    return ULMBLAS(sdot)(n, x, incX, y, incY);
}


double
CBLAS(ddot)(ULMBLAS_INDEX_TYPE n,
            const double       *x,
            ULMBLAS_INDEX_TYPE incX,
            const double       *y,
            ULMBLAS_INDEX_TYPE incY)
{
    return ULMBLAS(ddot)(n, x, incX, y, incY);
}

void
CBLAS(cdotc_sub)(ULMBLAS_INDEX_TYPE n,
                 const float        *x,
                 ULMBLAS_INDEX_TYPE incX,
                 const float        *y,
                 ULMBLAS_INDEX_TYPE incY,
                 float              *dotc)
{
    return ULMBLAS(cdotc_sub)(n, x, incX, y, incY, dotc);
}

void
CBLAS(zdotc_sub)(ULMBLAS_INDEX_TYPE n,
                 const double       *x,
                 ULMBLAS_INDEX_TYPE incX,
                 const double       *y,
                 ULMBLAS_INDEX_TYPE incY,
                 double             *dotc)
{
    return ULMBLAS(zdotc_sub)(n, x, incX, y, incY, dotc);
}

void
CBLAS(cdotu_sub)(ULMBLAS_INDEX_TYPE n,
                 const float        *x,
                 ULMBLAS_INDEX_TYPE incX,
                 const float        *y,
                 ULMBLAS_INDEX_TYPE incY,
                 float              *dotc)
{
    return ULMBLAS(cdotu_sub)(n, x, incX, y, incY, dotc);
}

void
CBLAS(zdotu_sub)(ULMBLAS_INDEX_TYPE n,
                 const double       *x,
                 ULMBLAS_INDEX_TYPE incX,
                 const double       *y,
                 ULMBLAS_INDEX_TYPE incY,
                 double             *dotc)
{
    return ULMBLAS(zdotu_sub)(n, x, incX, y, incY, dotc);
}

} // extern "C"
