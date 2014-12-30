#include BLAS_HEADER
#include <ulmblas/level1/copy.h>

extern "C" {

void
ULMBLAS(dcopy)(int          n,
               const double *x,
               int          incX,
               double       *y,
               int          incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::copy(n, x, incX, y, incY);
}

void
CBLAS(dcopy)(int          n,
             const double *x,
             int          incX,
             double       *y,
             int          incY)
{
    ULMBLAS(dcopy)(n, x, incX, y, incY);
}

} // extern "C"
