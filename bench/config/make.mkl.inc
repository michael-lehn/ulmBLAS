mkl_NAME           = "Intel MKL"

#
#  Comparison with Intel MKL (not used by default).  Set MKLROOT (as done by
#  the MKL environment scripts):
#
#      make bench USE_BLAS_VARIANTS="ulmBLAS mkl" MKLROOT=...
#
ifeq ($(MKLROOT),)
$(error the benchmark variant 'mkl' requires MKLROOT (MKL installation directory))
endif

ATLAS_BLAS_LIB_NAMES   = libatlblas.a libcblas.a
ATLAS_LAPACK_LIB_NAMES = libatllapack.a

REF_BLAS_LIBS = $(MKLROOT)/lib/libmkl_intel_lp64.a $(MKLROOT)/lib/libmkl_core.a $(MKLROOT)/lib/libmkl_sequential.a -lpthread -lm

GREP_OPTS        = -v PASS
