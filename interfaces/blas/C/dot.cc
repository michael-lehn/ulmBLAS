#include <interfaces/blas/C/config.h>
#include <ulmblas/level1/dot.h>

extern "C" {

double
ULMBLAS(ddot)(const int     n,
              const double  *x,
              const int     incX,
              const double  *y,
              const int     incY)
{
    return ulmBLAS::dotu(n, x, incX, y, incY);
}

} // extern "C"
