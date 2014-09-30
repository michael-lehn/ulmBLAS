#include <interfaces/F77/config.h>
#include <ulmblas/level1/scal.h>

extern "C" {

void
F77BLAS(dscal)(const int    *_n,
               const double *_alpha,
               double       *x,
               const int    *_incX)
{
//
//  Dereference scalar parameters
//
    int n        = *_n;
    double alpha = *_alpha;
    int incX     = *_incX;

    ulmBLAS::scal(n, alpha, x, incX);
}

} // extern "C"
