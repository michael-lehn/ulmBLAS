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
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::swap(n, x, incX, y, incY);
}

void
CBLAS(dswap)(const int n,
             double    *x,
             const int incX,
             double    *y,
             const int incY)
{
    ULMBLAS(dswap)(n, x, incX, y, incY);
}

} // extern "C"
