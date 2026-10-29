#
#  Test configuration s-blis_haswell_4x24: the vendored SGEMM micro kernel blis_haswell_4x24
#  (ulmblas/level3/ukernel/blis/select_s.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="s-blis_haswell_4x24 s-blis_haswell_4x24-small"
#
CONFIG_DESCRIPTION = SGEMM kernel BLIS haswell_4x24, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_4x24
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
