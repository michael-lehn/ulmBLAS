#include BLAS_HEADER
#include <ulmblas/level1/nrm2.h>

extern "C" {

double
ULMBLAS(dnrm2)(const int     n,
               const double  *x,
               const int     incX)
{
    return ulmBLAS::nrm2(n, x, incX);
}

double
CBLAS(dnrm2)(const int     n,
             const double  *x,
             const int     incX)
{
    return ulmBLAS::nrm2(n, x, incX);
}

} // extern "C"
