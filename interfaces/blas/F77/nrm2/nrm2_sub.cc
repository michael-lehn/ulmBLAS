#include <interfaces/blas/F77/config.h>
#include <ulmblas/level1/nrm2.h>

#include <stdio.h>

extern "C" {

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

} // extern "C"
