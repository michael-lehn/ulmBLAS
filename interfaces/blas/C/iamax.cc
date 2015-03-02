#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

int
ULMBLAS(idamax)(int           n,
                const double  *x,
                int           incX)
{
    return ulmBLAS::iamax(n, x, incX);
}

int
ULMBLAS(izamax)(int           n,
                const double  *x_,
                int           incX)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    return ulmBLAS::iamax(n, x, incX);
}


int
CBLAS(idamax)(int           n,
              const double  *x,
              int           incX)
{
    return ULMBLAS(idamax)(n, x, incX);
}

int
CBLAS(izamax)(int           n,
              const double  *x,
              int           incX)
{
    return ULMBLAS(izamax)(n, x, incX);
}

} // extern "C"
