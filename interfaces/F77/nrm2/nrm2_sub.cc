#include <interfaces/F77/config.h>
#include <ulmblas/level1/nrm2.h>

#include <stdio.h>

extern "C" {

void
F77BLAS(dnrm2_sub)(const int     *_n,
                   const double  *x,
                   const int     *_incX,
                   double        *_result)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;

    if (incX<0) {
        x -= incX*(n-1);
    }
    *_result = ulmBLAS::nrm2(n, x, incX);
}

} // extern "C"
