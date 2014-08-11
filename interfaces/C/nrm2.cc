#include <interfaces/C/config.h>
#include <src/level1/nrm2.h>
#include <src/level1/nrm2.tcc>

extern "C" {

double
ULMBLAS(dnrm2)(const int     n,
               const double  *x,
               const int     incX)
{
    return ulmBLAS::nrm2(n, x, incX);
}

} // extern "C"
