mkl_NAME           = "Intel MKL (2013, SP1)"

ATLAS_BLAS_LIB_NAMES   = libatlblas.a libcblas.a
ATLAS_LAPACK_LIB_NAMES = libatllapack.a

MKLROOT       = /opt/intel/composer_xe_2013_sp1.3.166/mkl/
REF_BLAS_LIBS = $(MKLROOT)/lib/libmkl_intel_lp64.a $(MKLROOT)/lib/libmkl_core.a $(MKLROOT)/lib/libmkl_sequential.a -lpthread -lm

MAKE_ATL_BLAS    = $(MAKE) -C ../interfaces/blas/C
MAKE_ATL_LAPACK  = $(MAKE) -C ../interfaces/lapack/C
MAKE_REF_BLAS    =

GREP_OPTS        = -v PASS


LDFLAGS += $(LDF77FLAGS)
LDLIBS  += -lgfortran


