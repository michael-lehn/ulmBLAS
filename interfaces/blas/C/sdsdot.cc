#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

float
ULMBLAS(sdsdot)(ULMBLAS_INDEX_TYPE n,
                float              alpha,
                const float        *x,
                ULMBLAS_INDEX_TYPE incX,
                const float        *y,
                ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    double result = alpha;
    ulmBLAS::sdsdot<std::ptrdiff_t>(n, x, incX, y, incY, result);
    return float(result);
}

double
ULMBLAS(dsdot)(ULMBLAS_INDEX_TYPE n,
               const float        *x,
               ULMBLAS_INDEX_TYPE incX,
               const float        *y,
               ULMBLAS_INDEX_TYPE incY)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    double result = 0;
    ulmBLAS::sdsdot<std::ptrdiff_t>(n, x, incX, y, incY, result);
    return result;
}

float
CBLAS(sdsdot)(ULMBLAS_INDEX_TYPE n,
              float              alpha,
              const float        *x,
              ULMBLAS_INDEX_TYPE incX,
              const float        *y,
              ULMBLAS_INDEX_TYPE incY)
{
    return ULMBLAS(sdsdot)(n, alpha, x, incX, y, incY);
}

double
CBLAS(dsdot)(ULMBLAS_INDEX_TYPE n,
             const float        *x,
             ULMBLAS_INDEX_TYPE incX,
             const float        *y,
             ULMBLAS_INDEX_TYPE incY)
{
    return ULMBLAS(dsdot)(n, x, incX, y, incY);
}

} // extern "C"
