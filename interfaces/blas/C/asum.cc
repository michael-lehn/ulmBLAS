#include BLAS_HEADER
#include <ulmblas/level1/asum.h>

extern "C" {

double
ULMBLAS(dasum)(const int     n,
               const double  *x,
               const int     incX)
{
    return ulmBLAS::asum(n, x, incX);
}

} // extern "C"
