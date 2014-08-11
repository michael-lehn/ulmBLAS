#include <interfaces/C/config.h>
#include <src/level1/dot.h>
#include <src/level1/dot.tcc>

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
