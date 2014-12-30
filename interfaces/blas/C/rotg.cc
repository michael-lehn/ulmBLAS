#include BLAS_HEADER
#include <ulmblas/level1/rot.h>

extern "C" {

void
ULMBLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
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
