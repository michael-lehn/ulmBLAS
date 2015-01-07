#include BLAS_HEADER
#include <complex>
#include <ulmblas/level1/dot.h>

extern "C" {

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
