CONFIG  = shared64

#  Fortran 77 interface with 64-bit integers (ILP64) and the suffix _64_
#  (dgemm_64_), the second interface of the shared library of this
#  directory (ILP64=1), as the reference BLAS built with
#  BUILD_INDEX64_EXT_API and OpenBLAS with INTERFACE64=1 SYMBOLSUFFIX=64_.
F77BLAS_SUFFIX        = _64_
F77BLAS_LIB_NAME      = lib$(CONFIG)blas.a
F77BLAS_HEADER_NAME   = $(CONFIG)blas.h
F77BLAS_INCLUDE_GUARD = ULMBLAS_$(shell echo $(CONFIG) | tr "a-z" "A-Z")BLAS_H
