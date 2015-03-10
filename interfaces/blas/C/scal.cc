#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sscal)(const int    n,
               const float  alpha,
               float        *x,
               const int    incX)
{
    ulmBLAS::scal(n, alpha, x, incX);
}

void
ULMBLAS(dscal)(const int    n,
               const double alpha,
               double       *x,
               const int    incX)
{
    ulmBLAS::scal(n, alpha, x, incX);
}

void
ULMBLAS(csscal)(const int    n,
                const float  alpha,
                float        *x_,
                const int    incX)
{
    typedef std::complex<float> fcomplex;
    fcomplex *x = reinterpret_cast<fcomplex *>(x_);

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
ULMBLAS(cscal)(const int    n,
               const float  *alpha_,
               float        *x_,
               const int    incX)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex *x    = reinterpret_cast<fcomplex *>(x_);

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
CBLAS(sscal)(const int    n,
             const float  alpha,
             float        *x,
             const int    incX)
{
    ULMBLAS(sscal)(n, alpha, x, incX);
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
CBLAS(csscal)(const int    n,
              const float  alpha,
              float        *x,
              const int    incX)
{
    ULMBLAS(csscal)(n, alpha, x, incX);
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
CBLAS(cscal)(const int    n,
             const float  *alpha,
             float        *x,
             const int    incX)
{
    ULMBLAS(cscal)(n, alpha, x, incX);
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
