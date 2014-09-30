#include <interfaces/F77/config.h>
#include <ulmblas/level1/asum.h>

extern "C" {

void
F77BLAS(dasum_sub)(const int    *_n,
                   const double *x,
                   const int    *_incX,
                   double       *_result)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;

    *_result = ulmBLAS::asum(n, x, incX);
}

} // extern "C"
