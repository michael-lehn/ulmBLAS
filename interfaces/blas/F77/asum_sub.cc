#include BLAS_HEADER
#include <complex>
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(sasum_sub)(const ULMBLAS_INDEX_TYPE *n_,
                   const float              *x,
                   const ULMBLAS_INDEX_TYPE *incX_,
                   float                    *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    *result_ = ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

void
F77BLAS(dasum_sub)(const ULMBLAS_INDEX_TYPE *n_,
                   const double             *x,
                   const ULMBLAS_INDEX_TYPE *incX_,
                   double                   *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    *result_ = ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

void
F77BLAS(scasum_sub)(const ULMBLAS_INDEX_TYPE *n_,
                    const float              *x_,
                    const ULMBLAS_INDEX_TYPE *incX_,
                    float                    *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<float> fcomplex;
    const fcomplex *x = reinterpret_cast<const fcomplex *>(x_);

    *result_ = ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

void
F77BLAS(dzasum_sub)(const ULMBLAS_INDEX_TYPE *n_,
                    const double             *x_,
                    const ULMBLAS_INDEX_TYPE *incX_,
                    double                   *result_)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;

    typedef std::complex<double> dcomplex;
    const dcomplex *x = reinterpret_cast<const dcomplex *>(x_);

    *result_ = ulmBLAS::asum<std::ptrdiff_t>(n, x, incX);
}

} // extern "C"
