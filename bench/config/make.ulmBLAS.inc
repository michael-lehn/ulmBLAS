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

MAKE_ATL_BLAS    = $(foreach target, atl cblas, \
                        $(MAKE) -C ../interfaces/blas/C $(target);)
MAKE_ATL_LAPACK  = $(MAKE) -C ../interfaces/lapack/C atl
MAKE_REF_BLAS    = $(MAKE) -C ../interfaces/blas/F77

GREP_OPTS        = PASS


LDFLAGS += $(LDF77FLAGS)
LDLIBS  += -lgfortran

