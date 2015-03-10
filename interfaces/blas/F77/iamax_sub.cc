#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(isamax_sub)(const int       *n_,
                    const float     *x,
                    const int       *incX_,
                    int             *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    *result_ = ulmBLAS::iamax(n, x, incX)+1;
}

void
F77BLAS(idamax_sub)(const int       *n_,
                    const double    *x,
                    const int       *incX_,
                    int             *result_)
{
//
//  Dereference scalar parameters
//
    int n    = *n_;
    int incX = *incX_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    *result_ = ulmBLAS::iamax(n, x, incX)+1;
}

void
F77BLAS(icamax_sub)(const int       *n_,
                    const float     *x_,
                    const int       *incX_,
                    int             *result_)
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
    *result_ = ulmBLAS::iamax(n, x, incX)+1;
}

void
F77BLAS(izamax_sub)(const int       *n_,
                    const double    *x_,
                    const int       *incX_,
                    int             *result_)
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
    *result_ = ulmBLAS::iamax(n, x, incX)+1;
}

} // extern "C"
