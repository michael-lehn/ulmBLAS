#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(dscal)(const int    n,
               const double alpha,
               double       *x,
               const int    incX)
{
    ulmBLAS::scal(n, alpha, x, incX);
}

void
ULMBLAS(zdscal)(const int    n,
                const double alpha,
                double       *x_,
                const int    incX)
{
    typedef std::complex<double> dcomplex;
    dcomplex *x = reinterpret_cast<dcomplex *>(x_);

    ulmBLAS::scal(n, alpha, x, incX);
}

void
ULMBLAS(zscal)(const int    n,
               const double *alpha_,
               double       *x_,
               const int    incX)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex *x    = reinterpret_cast<dcomplex *>(x_);

    ulmBLAS::scal(n, alpha, x, incX);
}

void
CBLAS(dscal)(const int    n,
             const double alpha,
             double       *x,
             const int    incX)
{
    ULMBLAS(dscal)(n, alpha, x, incX);
}

void
CBLAS(zdscal)(const int    n,
              const double alpha,
              double       *x,
              const int    incX)
{
    ULMBLAS(zdscal)(n, alpha, x, incX);
}

void
CBLAS(zscal)(const int    n,
             const double *alpha,
             double       *x,
             const int    incX)
{
    ULMBLAS(zscal)(n, alpha, x, incX);
}

} // extern "C"
