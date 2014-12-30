CBLAS_PREFIX        = cblas_
ULMBLAS_PREFIX      = ulmblas_
CBLAS_LIB_NAME      = lib$(CONFIG)cblas.a
CBLAS_HEADER_NAME   = $(CONFIG)cblas.h
CBLAS_INCLUDE_GUARD = ULMBLAS_$(shell echo $(CONFIG) | tr "a-z" "A-Z")CBLAS_H

SUMMARY_SUFFIX = default
