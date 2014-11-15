CONFIG  = atl


CBLAS_PREFIX        = $(shell echo $(CONFIG) | tr "a-z" "A-Z")_
CBLAS_LIB_NAME      = lib$(CONFIG)blas.a
CBLAS_HEADER_NAME   = $(CONFIG)blas.h
CBLAS_INCLUDE_GUARD = ULMBLAS_$(shell echo $(CONFIG) | tr "a-z" "A-Z")BLAS_H

CLAPACK_PREFIX      = $(CBLAS_PREFIX)
CLAPACK_LIB_NAME    = lib$(CONFIG)lapack.a
CLAPACK_HEADER_NAME = $(CONFIG)lapack.h
CBLAS_INCLUDE_GUARD = ULMBLAS_$(shell echo $(CONFIG) | tr "a-z" "A-Z")LAPACK_H
