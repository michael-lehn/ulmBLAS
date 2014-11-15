#ifndef INTERFACE_BLAS_F77_XERBLA_H
#define INTERFACE_BLAS_F77_XERBLA_H 1

extern "C" {

void
F77BLAS(xerbla)(const char rout[6], const int *info);

} // extern "C"

#endif // INTERFACE_BLAS_F77_XERBLA_H
