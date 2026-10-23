atlas_NAME         = "ATLAS"

#
#  Comparison with ATLAS (not used by default).  Set ATLAS_LIBDIR to the
#  directory containing libatlas.a, libcblas.a and liblapack.a of ATLAS:
#
#      make bench USE_BLAS_VARIANTS="ulmBLAS atlas" ATLAS_LIBDIR=...
#
ifeq ($(ATLAS_LIBDIR),)
$(error the benchmark variant 'atlas' requires ATLAS_LIBDIR (directory of the ATLAS libraries))
endif

ATLAS_BLAS_LIBS    = $(ATLAS_LIBDIR)/libatlas.a $(ATLAS_LIBDIR)/libcblas.a
ATLAS_LAPACK_LIBS  = $(ATLAS_LIBDIR)/liblapack.a

REF_BLAS_LIB_NAMES     = librefblas.a

MAKE_ATL_BLAS    =
MAKE_ATL_LAPACK  =
MAKE_REF_BLAS    = $(MAKE) -C $(TOPDIR) refblas

GREP_OPTS        = PASS
