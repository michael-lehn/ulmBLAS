#include BLAS_HEADER
#include <complex>
#include <ulmblas/level1/nrm2.h>

#include <iostream>

extern "C" {

double
ULMBLAS(dnrm2)(int           n,
               const double  *x,
               int           incX)
{
    return ulmBLAS::nrm2(n, x, incX);
}

double
ULMBLAS(dznrm2)(int           n,
                const double  *x_,
                int           incX)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    return ulmBLAS::nrm2(n, x, incX);
}

double
CBLAS(dnrm2)(int            n,
             const double   *x,
             int            incX)
{
    return ULMBLAS(dnrm2)(n, x, incX);
}

double
CBLAS(dznrm2)(int           n,
              const double  *x,
              int           incX)
{
    return ULMBLAS(dznrm2)(n, x, incX);
}

} // extern "C"
