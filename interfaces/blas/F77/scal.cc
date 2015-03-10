#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sscal)(const int    *n_,
               const float  *alpha_,
               float        *x,
               const int    *incX_)
{
//
//  Dereference scalar parameters
//
    int n        = *n_;
    float alpha = *alpha_;
    int incX     = *incX_;

    ulmBLAS::scal(n, alpha, x, incX);
}

void
F77BLAS(dscal)(const int    *n_,
               const double *alpha_,
               double       *x,
               const int    *incX_)
{
//
//  Dereference scalar parameters
//
    int n        = *n_;
    double alpha = *alpha_;
    int incX     = *incX_;

    ulmBLAS::scal(n, alpha, x, incX);
}

void
F77BLAS(cscal)(const int    *n_,
               const float  *alpha_,
               float        *x_,
               const int    *incX_)
{
//
//  Dereference scalar parameters
//
    int n        = *n_;
    int incX     = *incX_;

    typedef std::complex<float> fcomplex;
    fcomplex alpha = fcomplex(alpha_[0], alpha_[1]);
    fcomplex *x    = reinterpret_cast<fcomplex *>(x_);

    ulmBLAS::scal(n, alpha, x, incX);
}

void
F77BLAS(zscal)(const int    *n_,
               const double *alpha_,
               double       *x_,
               const int    *incX_)
{
//
//  Dereference scalar parameters
//
    int n        = *n_;
    int incX     = *incX_;

    typedef std::complex<double> dcomplex;
    dcomplex alpha = dcomplex(alpha_[0], alpha_[1]);
    dcomplex *x    = reinterpret_cast<dcomplex *>(x_);

    ulmBLAS::scal(n, alpha, x, incX);
}

void
F77BLAS(csscal)(const int    *n_,
                const float  *alpha_,
                float        *x_,
                const int    *incX_)
{
//
//  Dereference scalar parameters
//
    int n        = *n_;
    float  alpha = *alpha_;
    int incX     = *incX_;

    typedef std::complex<float> fcomplex;
    fcomplex *x    = reinterpret_cast<fcomplex *>(x_);

    ulmBLAS::scal(n, alpha, x, incX);
}

void
F77BLAS(zdscal)(const int    *n_,
                const double *alpha_,
                double       *x_,
                const int    *incX_)
{
//
//  Dereference scalar parameters
//
    int n        = *n_;
    double alpha = *alpha_;
    int incX     = *incX_;

    typedef std::complex<double> dcomplex;
    dcomplex *x    = reinterpret_cast<dcomplex *>(x_);

    ulmBLAS::scal(n, alpha, x, incX);
}

} // extern "C"
