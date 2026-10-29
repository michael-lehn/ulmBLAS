#
#  Test configuration s-openblas_haswell_8x4: the vendored SGEMM micro kernel openblas_haswell_8x4
#  (ulmblas/level3/ukernel/blis/select_s.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="s-openblas_haswell_8x4 s-openblas_haswell_8x4-small"
#
CONFIG_DESCRIPTION = SGEMM kernel OpenBLAS haswell_8x4, default block sizes
CONFIG_CPPFLAGS += -mavx -mavx2 -mfma -DULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4
CONFIG_REQUIRES  = avx avx2 fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
