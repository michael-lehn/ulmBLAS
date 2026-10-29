#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(srotm)(const ULMBLAS_INDEX_TYPE n,
               float                    *x,
               const ULMBLAS_INDEX_TYPE incX,
               float                    *y,
               const ULMBLAS_INDEX_TYPE incY,
               const float              *param)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rotm<std::ptrdiff_t>(n, x, incX, y, incY, param);
}

void
ULMBLAS(drotm)(const ULMBLAS_INDEX_TYPE n,
               double                   *x,
               const ULMBLAS_INDEX_TYPE incX,
               double                   *y,
               const ULMBLAS_INDEX_TYPE incY,
               const double             *param)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rotm<std::ptrdiff_t>(n, x, incX, y, incY, param);
}

void
CBLAS(srotm)(const ULMBLAS_INDEX_TYPE n,
             float                    *x,
             const ULMBLAS_INDEX_TYPE incX,
             float                    *y,
             const ULMBLAS_INDEX_TYPE incY,
             const float              *param)
{
    ULMBLAS(srotm)(n, x, incX, y, incY, param);
}

void
CBLAS(drotm)(const ULMBLAS_INDEX_TYPE n,
             double                   *x,
             const ULMBLAS_INDEX_TYPE incX,
             double                   *y,
             const ULMBLAS_INDEX_TYPE incY,
             const double             *param)
{
    ULMBLAS(drotm)(n, x, incX, y, incY, param);
}

} // extern "C"
