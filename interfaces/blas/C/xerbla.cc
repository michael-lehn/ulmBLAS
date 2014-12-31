#include <interfaces/blas/C/xerbla.h>
#include <cstdio>

extern "C" {

void
ULMBLAS(xerbla)(int info, const char *rout, const char *form, ...)
{
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", info, rout);
}

void
CBLAS(xerbla)(int info, const char *rout, const char *form, ...)
{
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", info, rout);
}

} // extern "C"
