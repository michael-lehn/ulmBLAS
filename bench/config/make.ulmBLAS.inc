ulmBLAS_NAME         = "ulmBLAS"

ATLAS_BLAS_LIB_NAMES   = libatlblas.a libcblas.a
ATLAS_LAPACK_LIB_NAMES = libatllapack.a

REF_BLAS_LIB_NAMES     = librefblas.a

MAKE_ATL_BLAS    = $(MAKE) -C ../interfaces/blas/C atl cblas
MAKE_ATL_LAPACK  = $(MAKE) -C ../interfaces/lapack/C atl
MAKE_REF_BLAS    = $(MAKE) -C .. refblas

GREP_OPTS        = PASS


LDFLAGS += $(LDF77FLAGS)
LDLIBS  += -lgfortran

