CONFIG  = cblas64

#  C interface with 64-bit integers (ILP64) and the suffix _64
#  (cblas_dgemm_64), the second interface of the shared library of this
#  directory (ILP64=1), as the reference CBLAS built with
#  BUILD_INDEX64_EXT_API.  The ulmblas_ functions behind it get the suffix
#  _ilp64 (not exported); the globals of the reference CBLAS
#  (RowMajorStrg, CBLAS_CallFromC) and the helper transpose are those of
#  the LP64 interface, the thread functions and GEMM with arbitrary
#  strides exist once (LP64).
CBLAS_PREFIX          = cblas_
CBLAS_SUFFIX          = _64
ULMBLAS_PREFIX        = ulmblas_
ULMBLAS_SUFFIX        = _ilp64
CBLAS_LIB_NAME        = lib$(CONFIG).a
CBLAS_HEADER_NAME     = $(CONFIG).h
CBLAS_INCLUDE_GUARD   = ULMBLAS_$(shell echo $(CONFIG) | tr "a-z" "A-Z")_H
EXCLUDED_SOURCES      = cblas_globals.cc transpose.cc threads.cc gemm_strided.cc
