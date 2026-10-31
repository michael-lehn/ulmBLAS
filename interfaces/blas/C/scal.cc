#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sscal)(const ULMBLAS_INDEX_TYPE n,
               const float              alpha,
               float                    *x,
               const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
ULMBLAS(dscal)(const ULMBLAS_INDEX_TYPE n,
               const double             alpha,
               double                   *x,
               const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
ULMBLAS(csscal)(const ULMBLAS_INDEX_TYPE n,
                const float              alpha,
                float                    *x_,
                const ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<float> fcomplex;
    fcomplex *x = reinterpret_cast<fcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}


void
ULMBLAS(zdscal)(const ULMBLAS_INDEX_TYPE n,
                const double             alpha,
                double                   *x_,
                const ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<double> dcomplex;
    dcomplex *x = reinterpret_cast<dcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
ULMBLAS(cscal)(const ULMBLAS_INDEX_TYPE n,
               const float              *alpha_,
               float                    *x_,
               const ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex *x    = reinterpret_cast<fcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}


void
ULMBLAS(zscal)(const ULMBLAS_INDEX_TYPE n,
               const double             *alpha_,
               double                   *x_,
               const ULMBLAS_INDEX_TYPE incX)
{
    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex *x    = reinterpret_cast<dcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
CBLAS(sscal)(const ULMBLAS_INDEX_TYPE n,
             const float              alpha,
             float                    *x,
             const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<=0) {
        return;
    }
    ULMBLAS(sscal)(n, alpha, x, incX);
}

void
CBLAS(dscal)(const ULMBLAS_INDEX_TYPE n,
             const double             alpha,
             double                   *x,
             const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<=0) {
        return;
    }
    ULMBLAS(dscal)(n, alpha, x, incX);
}

void
CBLAS(csscal)(const ULMBLAS_INDEX_TYPE n,
              const float              alpha,
              float                    *x,
              const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<=0) {
        return;
    }
    ULMBLAS(csscal)(n, alpha, x, incX);
}

void
CBLAS(zdscal)(const ULMBLAS_INDEX_TYPE n,
              const double             alpha,
              double                   *x,
              const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<=0) {
        return;
    }
    ULMBLAS(zdscal)(n, alpha, x, incX);
}

void
CBLAS(cscal)(const ULMBLAS_INDEX_TYPE n,
             const float              *alpha,
             float                    *x,
             const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<=0) {
        return;
    }
    ULMBLAS(cscal)(n, alpha, x, incX);
}

void
CBLAS(zscal)(const ULMBLAS_INDEX_TYPE n,
             const double             *alpha,
             double                   *x,
             const ULMBLAS_INDEX_TYPE incX)
{
    if (incX<=0) {
        return;
    }
    ULMBLAS(zscal)(n, alpha, x, incX);
}

} // extern "C"
