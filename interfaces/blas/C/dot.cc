#include BLAS_HEADER
#include <ulmblas/level1/dot.h>

extern "C" {

double
ULMBLAS(ddot)(int           n,
              const double  *x,
              int           incX,
              const double  *y,
              int           incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    return ulmBLAS::dotu(n, x, incX, y, incY);
}

double
CBLAS(ddot)(int           n,
            const double  *x,
            int           incX,
            const double  *y,
            int           incY)
{
    return ULMBLAS(ddot)(n, x, incX, y, incY);
}

} // extern "C"
