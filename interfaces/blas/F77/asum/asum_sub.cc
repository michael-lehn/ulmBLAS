#include <interfaces/blas/F77/config.h>
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

} // extern "C"
