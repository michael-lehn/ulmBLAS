#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

int
ULMBLAS(isamax)(int           n,
                const float   *x,
                int           incX)
{
    return ulmBLAS::iamax(n, x, incX);
}


int
ULMBLAS(idamax)(int           n,
                const double  *x,
                int           incX)
{
    return ulmBLAS::iamax(n, x, incX);
}

int
ULMBLAS(icamax)(int           n,
                const float   *x_,
                int           incX)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);

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
CBLAS(isamax)(int           n,
              const float   *x,
              int           incX)
{
    return ULMBLAS(isamax)(n, x, incX);
}

int
CBLAS(idamax)(int           n,
              const double  *x,
              int           incX)
{
    return ULMBLAS(idamax)(n, x, incX);
}

int
CBLAS(icamax)(int           n,
              const float   *x,
              int           incX)
{
    return ULMBLAS(icamax)(n, x, incX);
}

int
CBLAS(izamax)(int           n,
              const double  *x,
              int           incX)
{
    return ULMBLAS(izamax)(n, x, incX);
}

} // extern "C"
