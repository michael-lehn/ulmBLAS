#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

float
ULMBLAS(sdot)(int           n,
              const float   *x,
              int           incX,
              const float   *y,
              int           incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::dotu(n, x, incX, y, incY);
}

double
ULMBLAS(ddot)(int           n,
              const double  *x,
              int           incX,
              const double  *y,
              int           incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::dotu(n, x, incX, y, incY);
}

void
ULMBLAS(cdotc_sub)(int           n,
                   const float   *x_,
                   int           incX,
                   const float   *y_,
                   int           incY,
                   float         *dotc_)
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

    fcomplex dotc = ulmBLAS::dotc(n, x, incX, y, incY);
    dotc_[0] = std::real(dotc);
    dotc_[1] = std::imag(dotc);
}

void
ULMBLAS(zdotc_sub)(int           n,
                   const double  *x_,
                   int           incX,
                   const double  *y_,
                   int           incY,
                   double        *dotc_)
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

    dcomplex dotc = ulmBLAS::dotc(n, x, incX, y, incY);
    dotc_[0] = std::real(dotc);
    dotc_[1] = std::imag(dotc);
}

void
ULMBLAS(cdotu_sub)(int           n,
                   const float   *x_,
                   int           incX,
                   const float   *y_,
                   int           incY,
                   float         *dotu_)
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

    fcomplex dotu = ulmBLAS::dotu(n, x, incX, y, incY);
    dotu_[0] = std::real(dotu);
    dotu_[1] = std::imag(dotu);
}

void
ULMBLAS(zdotu_sub)(int           n,
                   const double  *x_,
                   int           incX,
                   const double  *y_,
                   int           incY,
                   double        *dotu_)
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

    dcomplex dotu = ulmBLAS::dotu(n, x, incX, y, incY);
    dotu_[0] = std::real(dotu);
    dotu_[1] = std::imag(dotu);
}

float
CBLAS(sdot)(int           n,
            const float   *x,
            int           incX,
            const float   *y,
            int           incY)
{
    return ULMBLAS(sdot)(n, x, incX, y, incY);
}


double
CBLAS(ddot)(int           n,
            const double  *x,
            int           incX,
            const double  *y,
            int           incY)
{
    return ULMBLAS(ddot)(n, x, incX, y, incY);
}

void
CBLAS(cdotc_sub)(int           n,
                 const float   *x,
                 int           incX,
                 const float   *y,
                 int           incY,
                 float         *dotc)
{
    return ULMBLAS(cdotc_sub)(n, x, incX, y, incY, dotc);
}

void
CBLAS(zdotc_sub)(int           n,
                 const double  *x,
                 int           incX,
                 const double  *y,
                 int           incY,
                 double        *dotc)
{
    return ULMBLAS(zdotc_sub)(n, x, incX, y, incY, dotc);
}

void
CBLAS(cdotu_sub)(int           n,
                 const float   *x,
                 int           incX,
                 const float   *y,
                 int           incY,
                 float         *dotc)
{
    return ULMBLAS(cdotu_sub)(n, x, incX, y, incY, dotc);
}

void
CBLAS(zdotu_sub)(int           n,
                 const double  *x,
                 int           incX,
                 const double  *y,
                 int           incY,
                 double        *dotc)
{
    return ULMBLAS(zdotu_sub)(n, x, incX, y, incY, dotc);
}

} // extern "C"
