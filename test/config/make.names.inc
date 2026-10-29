#
#  Names of the libraries and headers of a test configuration (each
#  configuration is built in its own directory).
#
CBLAS_PREFIX          = cblas_
ULMBLAS_PREFIX        = ulmblas_
CBLAS_LIB_NAME        = libcblas.a
CBLAS_HEADER_NAME     = cblas.h
CBLAS_INCLUDE_GUARD   = ULMBLAS_CBLAS_H

CLAPACK_PREFIX        = cblas_
ULMLAPACK_PREFIX      = ulmblas_
CLAPACK_LIB_NAME      = libclapack.a
CLAPACK_HEADER_NAME   = clapack.h
CLAPACK_INCLUDE_GUARD = ULMBLAS_CLAPACK_H

F77BLAS_SUFFIX        = _
F77BLAS_LIB_NAME      = libf77blas.a
F77BLAS_HEADER_NAME   = f77blas.h
F77BLAS_INCLUDE_GUARD = ULMBLAS_F77BLAS_H
