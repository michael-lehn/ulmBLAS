#
#  Test configuration z-openblas_haswell_4x2: the vendored ZGEMM micro kernel openblas_haswell_4x2
#  (ulmblas/level3/ukernel/blis/select_z.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="z-openblas_haswell_4x2 z-openblas_haswell_4x2-small"
#
CONFIG_DESCRIPTION = ZGEMM kernel OpenBLAS haswell_4x2, default block sizes
CONFIG_CPPFLAGS += -mavx -mavx2 -mfma -DULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2
CONFIG_REQUIRES  = avx avx2 fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
