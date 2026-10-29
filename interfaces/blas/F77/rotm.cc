#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

void
F77BLAS(srotm)(const ULMBLAS_INDEX_TYPE *n_,
               float                    *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               float                    *y,
               const ULMBLAS_INDEX_TYPE *incY_,
               const float              *param)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rotm<std::ptrdiff_t>(n, x, incX, y, incY, param);
}

void
F77BLAS(drotm)(const ULMBLAS_INDEX_TYPE *n_,
               double                   *x,
               const ULMBLAS_INDEX_TYPE *incX_,
               double                   *y,
               const ULMBLAS_INDEX_TYPE *incY_,
               const double             *param)
{
//
//  Dereference scalar parameters
//
    ULMBLAS_INDEX_TYPE n    = *n_;
    ULMBLAS_INDEX_TYPE incX = *incX_;
    ULMBLAS_INDEX_TYPE incY = *incY_;

    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rotm<std::ptrdiff_t>(n, x, incX, y, incY, param);
}

void
F77BLAS(srotmg)(float          *d1,
                float          *d2,
                float          *x1,
                const float    *y1,
                float          *param)
{
    ulmBLAS::rotmg(*d1, *d2, *x1, *y1, param);
}

void
F77BLAS(drotmg)(double         *d1,
                double         *d2,
                double         *x1,
                const double   *y1,
                double         *param)
{
    ulmBLAS::rotmg(*d1, *d2, *x1, *y1, param);
}

} // extern "C"
