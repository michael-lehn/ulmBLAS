#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(srotmg)(float          *d1,
                float          *d2,
                float          *x1,
                const float    y1,
                float          *param)
{
    ulmBLAS::rotmg(*d1, *d2, *x1, y1, param);
}

void
ULMBLAS(drotmg)(double         *d1,
                double         *d2,
                double         *x1,
                const double   y1,
                double         *param)
{
    ulmBLAS::rotmg(*d1, *d2, *x1, y1, param);
}

void
CBLAS(srotmg)(float          *d1,
              float          *d2,
              float          *x1,
              const float    y1,
              float          *param)
{
    ULMBLAS(srotmg)(d1, d2, x1, y1, param);
}

void
CBLAS(drotmg)(double         *d1,
              double         *d2,
              double         *x1,
              const double   y1,
              double         *param)
{
    ULMBLAS(drotmg)(d1, d2, x1, y1, param);
}

} // extern "C"
