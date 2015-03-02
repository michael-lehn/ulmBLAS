#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

double
ULMBLAS(dasum)(int           n,
               const double  *x,
               int           incX)
{
    return ulmBLAS::asum(n, x, incX);
}

double
ULMBLAS(dzasum)(int           n,
                const double  *x_,
                int           incX)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    return ulmBLAS::asum(n, x, incX);
}

double
CBLAS(dasum)(int           n,
             const double  *x,
             int           incX)
{
    return ulmBLAS::asum(n, x, incX);
}

double
CBLAS(dzasum)(int           n,
              const double  *x,
              int           incX)
{
    return ULMBLAS(dzasum)(n, x, incX);
}

} // extern "C"
