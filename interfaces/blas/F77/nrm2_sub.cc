#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(snrm2_sub)(const int     *n_,
                   const float   *x,
                   const int     *incX_,
                   float         *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    *result_ = ulmBLAS::nrm2(n, x, incX);
}

void
F77BLAS(dnrm2_sub)(const int     *n_,
                   const double  *x,
                   const int     *incX_,
                   double        *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    *result_ = ulmBLAS::nrm2(n, x, incX);
}

void
F77BLAS(scnrm2_sub)(const int     *n_,
                    const float   *x_,
                    const int     *incX_,
                    float         *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    *result_ = ulmBLAS::nrm2(n, x, incX);
}

void
F77BLAS(dznrm2_sub)(const int     *n_,
                    const double  *x_,
                    const int     *incX_,
                    double        *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    if (incX<0) {
        x -= incX*(n-1);
    }
    *result_ = ulmBLAS::nrm2(n, x, incX);
}

} // extern "C"
