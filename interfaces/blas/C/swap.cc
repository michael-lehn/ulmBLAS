#include BLAS_HEADER
#include <ulmblas/level1/swap.h>

extern "C" {


void
ULMBLAS(dswap)(const int n,
               double    *x,
               const int incX,
               double    *y,
               const int incY)
{
    return ulmBLAS::swap(n, x, incX, y, incY);
}

} // extern "C"
