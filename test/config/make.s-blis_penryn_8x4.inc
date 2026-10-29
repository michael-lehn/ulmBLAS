#
#  Test configuration s-blis_penryn_8x4: the vendored SGEMM micro kernel blis_penryn_8x4
#  (ulmblas/level3/ukernel/blis/select_s.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="s-blis_penryn_8x4 s-blis_penryn_8x4-small"
#
CONFIG_DESCRIPTION = SGEMM kernel BLIS penryn_8x4, default block sizes
CONFIG_CPPFLAGS += -msse3 -DULMBLAS_SGEMM_UKERNEL_blis_penryn_8x4
CONFIG_REQUIRES  = sse3

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
