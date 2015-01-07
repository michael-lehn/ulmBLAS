#include BLAS_HEADER
#include <complex>
#include <ulmblas/level1/asum.h>

extern "C" {

void
F77BLAS(dasum_sub)(const int    *n_,
                   const double *x,
                   const int    *incX_,
                   double       *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    *result_ = ulmBLAS::asum(n, x, incX);
}

void
F77BLAS(dzasum_sub)(const int    *n_,
                    const double *x_,
                    const int    *incX_,
                    double       *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    *result_ = ulmBLAS::asum(n, x, incX);
}

} // extern "C"
