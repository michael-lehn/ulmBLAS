#include <interfaces/C/config.h>
#include <src/level1/rot.h>

extern "C" {

void
ULMBLAS(drot)(const int      n,
              double         *x,
              const int      incX,
              double         *y,
              const int      incY,
              const double   c,
              const double   s)
{
    return ulmBLAS::rot(n, x, incX, y, incY, c, s);
}

} // extern "C"
