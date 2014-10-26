#include <interfaces/blas/F77/xerbla.h>
#include <cstdio>

extern "C" {

void
F77BLAS(xerbla)(const char rout[6], const int *info)
{
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", *info, rout);
}

} // extern "C"
