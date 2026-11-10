CONFIG  = shared

#  Fortran 77 interface with 32-bit integers (LP64) and the usual symbol
#  names (dgemm_), for the shared library of this directory.
F77BLAS_SUFFIX        = _
F77BLAS_LIB_NAME      = lib$(CONFIG)blas.a
F77BLAS_HEADER_NAME   = $(CONFIG)blas.h
F77BLAS_INCLUDE_GUARD = ULMBLAS_$(shell echo $(CONFIG) | tr "a-z" "A-Z")BLAS_H
