#ifndef INTERFACE_BLAS_C_TRANSPOSE_H
#define INTERFACE_BLAS_C_TRANSPOSE_H 1

#include BLAS_HEADER

extern "C" {

enum CBLAS_TRANSPOSE
transpose(enum CBLAS_TRANSPOSE trans);

} // extern "C"

#endif // INTERFACE_BLAS_C_TRANSPOSE_H
