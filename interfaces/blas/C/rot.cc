#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

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
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot(n, x, incX, y, incY, c, s);
}

void
CBLAS(drot)(const int      n,
            double         *x,
            const int      incX,
            double         *y,
            const int      incY,
            const double   c,
            const double   s)
{
    ULMBLAS(drot)(n, x, incX, y, incY, c, s);
}

} // extern "C"
