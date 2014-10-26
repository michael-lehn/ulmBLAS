#include <interfaces/blas/C/config.h>
#include <ulmblas/level1/axpy.h>

extern "C" {

void
ULMBLAS(daxpy)(const int     n,
               const double  alpha,
               const double  *x,
               const int     incX,
               double        *y,
               int           incY)
{
    return ulmBLAS::axpy(n, alpha, x, incX, y, incY);
}

} // extern "C"
