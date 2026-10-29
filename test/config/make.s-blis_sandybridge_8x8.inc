#
#  Test configuration s-blis_sandybridge_8x8: the vendored SGEMM micro kernel blis_sandybridge_8x8
#  (ulmblas/level3/ukernel/blis/select_s.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="s-blis_sandybridge_8x8 s-blis_sandybridge_8x8-small"
#
CONFIG_DESCRIPTION = SGEMM kernel BLIS sandybridge_8x8, default block sizes
CONFIG_CPPFLAGS += -mavx -DULMBLAS_SGEMM_UKERNEL_blis_sandybridge_8x8
CONFIG_REQUIRES  = avx

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
