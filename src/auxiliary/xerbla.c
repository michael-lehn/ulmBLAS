#include <auxiliary/xerbla.h>
#include <stdio.h>

//
//  Declaration for the BLAS error message function used and required by
//  BLAS level 2 and level 3 functions.
//
void
ULMBLAS(xerbla)(const char rout[6], const int info)
{
    fprintf(stderr, "Parameter %d to routine %s was incorrect\n", info, rout);
}

void
F77BLAS(xerbla)(const char rout[6], const int *info)
{
    ULMBLAS(xerbla)(rout, *info);
}
