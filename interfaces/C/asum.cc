#include <interfaces/C/config.h>
#include <src/level1/asum.h>

extern "C" {

double
ULMBLAS(dasum)(const int     n,
               const double  *x,
               const int     incX)
{
    return ulmBLAS::asum(n, x, incX);
}

} // extern "C"
