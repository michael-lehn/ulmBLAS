#
#  Test configuration c-blis_sandybridge_8x4: the vendored CGEMM micro kernel blis_sandybridge_8x4
#  (ulmblas/level3/ukernel/blis/select_c.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="c-blis_sandybridge_8x4 c-blis_sandybridge_8x4-small"
#
CONFIG_DESCRIPTION = CGEMM kernel BLIS sandybridge_8x4, default block sizes
CONFIG_CPPFLAGS += -mavx -DULMBLAS_CGEMM_UKERNEL_blis_sandybridge_8x4
CONFIG_REQUIRES  = avx

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
