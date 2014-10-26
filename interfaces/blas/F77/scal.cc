#include <interfaces/blas/F77/config.h>
#include <ulmblas/level1/scal.h>

extern "C" {

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

} // extern "C"
