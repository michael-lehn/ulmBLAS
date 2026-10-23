netlib_NAME      = "REFBLAS (Netlib)"

ATLAS_BLAS_LIB_NAMES   = libatlblas.a libcblas.a
ATLAS_LAPACK_LIB_NAMES = libatllapack.a

REF_BLAS_LIB_NAMES     = librefblas.a

MAKE_ATL_BLAS    = $(MAKE) -C $(TOPDIR) atlblas cblas
MAKE_ATL_LAPACK  = $(MAKE) -C $(TOPDIR) atllapack
MAKE_REF_BLAS    = $(MAKE) -C $(TOPDIR) refblas

GREP_OPTS        = -v PASS
