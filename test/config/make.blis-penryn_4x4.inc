#
#  Test configuration blis-penryn_4x4: the BLIS-derived DGEMM micro kernel
#  penryn_4x4 (ulmblas/level3/ukernel/blis/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="blis-penryn_4x4 blis-penryn_4x4-small"
#
CONFIG_DESCRIPTION = BLIS kernel penryn_4x4, default block sizes
CONFIG_CPPFLAGS += -msse3 -DULMBLAS_DGEMM_UKERNEL_blis_penryn_4x4
CONFIG_REQUIRES  = sse3

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
