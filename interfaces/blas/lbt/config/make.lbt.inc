CONFIG  = lbt

#  Fortran 77 interface with 64-bit integers for libblastrampoline (Julia):
#  symbols with the suffix _64_ (dgemm_64_), as in OpenBLAS's ILP64 build.
F77BLAS_SUFFIX        = _64_
F77BLAS_LIB_NAME      = lib$(CONFIG)blas.a
F77BLAS_HEADER_NAME   = $(CONFIG)blas.h
F77BLAS_INCLUDE_GUARD = ULMBLAS_$(shell echo $(CONFIG) | tr "a-z" "A-Z")BLAS_H
