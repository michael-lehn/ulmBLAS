#ifndef AUXILARY_XERBLA_H
#define AUXILARY_XERBLA_H

#include <ulmblas.h>

void
ULMBLAS(xerbla)(const char rout[6], const int info);

void
F77BLAS(xerbla)(const char rout[6], const int *info);

#endif // AUXILARY_XERBLA_H
