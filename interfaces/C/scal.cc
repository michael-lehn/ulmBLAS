#include <interfaces/C/config.h>
#include <src/level1/scal.h>

extern "C" {

void
ULMBLAS(dscal)(const int    n,
               const double alpha,
               double       *x,
               const int    incX)
{
    return ulmBLAS::scal(n, alpha, x, incX);
}

} // extern "C"
