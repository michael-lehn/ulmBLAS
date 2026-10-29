#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

ULMBLAS_INDEX_TYPE
ULMBLAS(isamax)(ULMBLAS_INDEX_TYPE n,
                const float        *x,
                ULMBLAS_INDEX_TYPE incX)
{
    return ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX);
}


ULMBLAS_INDEX_TYPE
ULMBLAS(idamax)(ULMBLAS_INDEX_TYPE n,
                const double       *x,
                ULMBLAS_INDEX_TYPE incX)
{
    return ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX);
}

ULMBLAS_INDEX_TYPE
ULMBLAS(icamax)(ULMBLAS_INDEX_TYPE n,
                const float        *x_,
                ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);

    return ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX);
}

ULMBLAS_INDEX_TYPE
ULMBLAS(izamax)(ULMBLAS_INDEX_TYPE n,
                const double       *x_,
                ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    return ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX);
}


ULMBLAS_INDEX_TYPE
CBLAS(isamax)(ULMBLAS_INDEX_TYPE n,
              const float        *x,
              ULMBLAS_INDEX_TYPE incX)
{
    return ULMBLAS(isamax)(n, x, incX);
}

ULMBLAS_INDEX_TYPE
CBLAS(idamax)(ULMBLAS_INDEX_TYPE n,
              const double       *x,
              ULMBLAS_INDEX_TYPE incX)
{
    return ULMBLAS(idamax)(n, x, incX);
}

ULMBLAS_INDEX_TYPE
CBLAS(icamax)(ULMBLAS_INDEX_TYPE n,
              const float        *x,
              ULMBLAS_INDEX_TYPE incX)
{
    return ULMBLAS(icamax)(n, x, incX);
}

ULMBLAS_INDEX_TYPE
CBLAS(izamax)(ULMBLAS_INDEX_TYPE n,
              const double       *x,
              ULMBLAS_INDEX_TYPE incX)
{
    return ULMBLAS(izamax)(n, x, incX);
}

} // extern "C"
