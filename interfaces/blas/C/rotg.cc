#include BLAS_HEADER
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(srotg)(float *a,
               float *b,
               float *c,
               float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
ULMBLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(srotg)(float *a,
             float *b,
             float *c,
             float *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

void
CBLAS(drotg)(double *a,
             double *b,
             double *c,
             double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

} // extern "C"
