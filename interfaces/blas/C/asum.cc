#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

float
ULMBLAS(sasum)(ULMBLAS_INDEX_TYPE n,
               const float        *x,
               ULMBLAS_INDEX_TYPE incX)
{
    return ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

double
ULMBLAS(dasum)(ULMBLAS_INDEX_TYPE n,
               const double       *x,
               ULMBLAS_INDEX_TYPE incX)
{
    return ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

float
ULMBLAS(scasum)(ULMBLAS_INDEX_TYPE n,
                const float        *x_,
                ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);

    return ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

double
ULMBLAS(dzasum)(ULMBLAS_INDEX_TYPE n,
                const double       *x_,
                ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    return ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

float
CBLAS(sasum)(ULMBLAS_INDEX_TYPE n,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX)
{
    return ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

double
CBLAS(dasum)(ULMBLAS_INDEX_TYPE n,
             const double       *x,
             ULMBLAS_INDEX_TYPE incX)
{
    return ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

float
CBLAS(scasum)(ULMBLAS_INDEX_TYPE n,
              const float        *x,
              ULMBLAS_INDEX_TYPE incX)
{
    return ULMBLAS(scasum)(n, x, incX);
}

double
CBLAS(dzasum)(ULMBLAS_INDEX_TYPE n,
              const double       *x,
              ULMBLAS_INDEX_TYPE incX)
{
    return ULMBLAS(dzasum)(n, x, incX);
}

} // extern "C"
