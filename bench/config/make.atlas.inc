atlas_NAME         = "ATLAS 3.10.1"

ATLAS_BLAS_LIBS    = $(HOME)/work/ATLAS/build/lib/libatlas.a $(HOME)/work/ATLAS/build/lib/libcblas.a
ATLAS_LAPACK_LIBS  = $(HOME)/work/ATLAS/build/lib/liblapack.a

REF_BLAS_LIB_NAMES     = librefblas.a

MAKE_ATL_BLAS    =
MAKE_ATL_LAPACK  =
MAKE_REF_BLAS    = $(MAKE) -C .. refblas

GREP_OPTS        = PASS


LDFLAGS += $(LDF77FLAGS)
LDLIBS  += -lgfortran


