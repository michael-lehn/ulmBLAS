#ifndef INTERFACE_BLAS_C_XERBLA_H
#define INTERFACE_BLAS_C_XERBLA_H 1

extern "C" {

void ULMBLAS(xerbla)(int info, const char *rout, const char *form, ...);

void CBLAS(xerbla)(ULMBLAS_INDEX_TYPE info, const char *rout, const char *form,
                   ...);

} // extern "C"

#endif // INTERFACE_BLAS_C_XERBLA_H
