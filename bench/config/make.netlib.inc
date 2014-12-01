netlib_NAME      = "REFBLAS (Netlib)"

ATLAS_BLAS_LIB_NAMES   = libatlblas.a libcblas.a
ATLAS_LAPACK_LIB_NAMES = libatllapack.a

REF_BLAS_LIB_NAMES     = librefblas.a

MAKE_ATL_BLAS    = $(foreach target, atl cblas, \
                        $(MAKE) -C ../interfaces/blas/C $(target);)
MAKE_ATL_LAPACK  = $(MAKE) -C ../interfaces/lapack/C atl
MAKE_REF_BLAS    = $(MAKE) -C .. refblas

GREP_OPTS        = -v PASS

LDFLAGS += $(LDF77FLAGS)
LDLIBS  += -lgfortran
