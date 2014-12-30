#include BLAS_HEADER
#include <ulmblas/level1/asum.h>

extern "C" {

double
ULMBLAS(dasum)(int           n,
               const double  *x,
               int           incX)
{
    return ulmBLAS::asum(n, x, incX);
}

double
CBLAS(dasum)(int           n,
             const double  *x,
             int           incX)
{
    return ulmBLAS::asum(n, x, incX);
}

} // extern "C"
