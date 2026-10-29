#
#  Test configuration openblas-haswell_4x8-kc600: the OpenBLAS-derived DGEMM micro kernel
#  haswell_4x8 with KC = 600, more than the kernel accepts per call (KMAX = 256): the
#  adapter splits the k loop of a tile into several calls, and the panel calls of the
#  macro kernel fall back to tiles (ulmblas/level3/ukernel/blis/ugemm.h).  See
#  make.openblas-haswell_4x8.inc.
#
CONFIG_DESCRIPTION = OpenBLAS kernel haswell_4x8, KC = 600 (k split in the adapter)
CONFIG_CPPFLAGS += -mavx -mavx2 -mfma -DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8
CONFIG_CPPFLAGS += -DULMBLAS_DGEMM_KC=600
CONFIG_REQUIRES  = avx avx2 fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
