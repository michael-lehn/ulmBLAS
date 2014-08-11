#include <interfaces/C/config.h>
#include <src/level1/rot.h>
#include <src/level1/rot.tcc>

extern "C" {

void
ULMBLAS(drotg)(double *a,
               double *b,
               double *c,
               double *s)
{
    ulmBLAS::rotg(*a, *b, *c, *s);
}

} // extern "C"
