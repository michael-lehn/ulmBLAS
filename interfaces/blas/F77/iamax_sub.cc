#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(isamax_sub)(const ULMBLAS_INDEX_TYPE *n_,
                    const float              *x,
                    const ULMBLAS_INDEX_TYPE *incX_,
                    ULMBLAS_INDEX_TYPE       *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    if (n<1 || incX<=0) {
        *result_ = 0;
        return;
    }
    *result_ = ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX)+1;
}

void
F77BLAS(idamax_sub)(const ULMBLAS_INDEX_TYPE *n_,
                    const double             *x,
                    const ULMBLAS_INDEX_TYPE *incX_,
                    ULMBLAS_INDEX_TYPE       *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    if (n<1 || incX<=0) {
        *result_ = 0;
        return;
    }
    *result_ = ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX)+1;
}

void
F77BLAS(icamax_sub)(const ULMBLAS_INDEX_TYPE *n_,
                    const float              *x_,
                    const ULMBLAS_INDEX_TYPE *incX_,
                    ULMBLAS_INDEX_TYPE       *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);

    if (n<1 || incX<=0) {
        *result_ = 0;
        return;
    }
    *result_ = ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX)+1;
}

void
F77BLAS(izamax_sub)(const ULMBLAS_INDEX_TYPE *n_,
                    const double             *x_,
                    const ULMBLAS_INDEX_TYPE *incX_,
                    ULMBLAS_INDEX_TYPE       *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    if (n<1 || incX<=0) {
        *result_ = 0;
        return;
    }
    *result_ = ulmBLAS::iamax<std::ptrdiff_t>(n, x, incX)+1;
}

} // extern "C"
