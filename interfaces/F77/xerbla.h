#ifndef INTERFACE_F77_XERBLA_H
#define INTERFACE_F77_XERBLA_H 1

#include <interfaces/F77/config.h>

extern "C" {

void
F77BLAS(xerbla)(const char rout[6], const int *info);

} // extern "C"

#endif // INTERFACE_F77_XERBLA_H
