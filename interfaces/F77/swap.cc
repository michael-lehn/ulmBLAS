#include <interfaces/F77/config.h>
#include <src/level1/swap.h>
#include <src/level1/swap.tcc>

extern "C" {

void
F77BLAS(dswap)(const int *_n,
               double    *x,
               const int *_incX,
               double    *y,
               const int *_incY)
{
//
//  Dereference scalar parameters
//
    int n    = *_n;
    int incX = *_incX;
    int incY = *_incY;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::swap(n, x, incX, y, incY);
}

} // extern "C"
