#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sscal)(const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               float                    *x,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n        = *n_;
    float alpha = *alpha_;
    ULMBLAS_INDEX_TYPE incX     = *incX_;

    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
F77BLAS(dscal)(const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               double                   *x,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    double alpha            = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
F77BLAS(cscal)(const ULMBLAS_INDEX_TYPE *n_,
               const float              *alpha_,
               float                    *x_,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex *x    = reinterpret_cast<fcomplex *>(x_);

    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
F77BLAS(zscal)(const ULMBLAS_INDEX_TYPE *n_,
               const double             *alpha_,
               double                   *x_,
               const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex *x    = reinterpret_cast<dcomplex *>(x_);

    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
F77BLAS(csscal)(const ULMBLAS_INDEX_TYPE *n_,
                const float              *alpha_,
                float                    *x_,
                const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n        = *n_;
    float  alpha = *alpha_;
    ULMBLAS_INDEX_TYPE incX     = *incX_;

    typedef std::complex<float> fcomplex;
    fcomplex *x    = reinterpret_cast<fcomplex *>(x_);

    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

void
F77BLAS(zdscal)(const ULMBLAS_INDEX_TYPE *n_,
                const double             *alpha_,
                double                   *x_,
                const ULMBLAS_INDEX_TYPE *incX_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    double alpha            = *alpha_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<double> dcomplex;
    dcomplex *x    = reinterpret_cast<dcomplex *>(x_);

    ulmBLAS::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

} // extern "C"
