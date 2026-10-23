ulmBLAS_NAME         = "ulmBLAS"

#
#  For speeding up benchmarks we compare the performance of ulmBLAS
#  with itself.
#
#  For comparing ulmBLAS with other BLAS libraries add a make.?.inc to
#  the bench/config/ directory.
#

ATLAS_BLAS_LIB_NAMES   = libatlblas.a libcblas.a
ATLAS_LAPACK_LIB_NAMES = libatllapack.a

REF_BLAS_LIB_NAMES     = libf77blas.a

MAKE_ATL_BLAS    = $(MAKE) -C $(TOPDIR) atlblas cblas
MAKE_ATL_LAPACK  = $(MAKE) -C $(TOPDIR) atllapack
MAKE_REF_BLAS    = $(MAKE) -C $(TOPDIR) f77blas

GREP_OPTS        = PASS
