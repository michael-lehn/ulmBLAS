#
#  Test configuration z-blis_sandybridge_4x4: the vendored ZGEMM micro kernel blis_sandybridge_4x4
#  (ulmblas/level3/ukernel/blis/select_z.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="z-blis_sandybridge_4x4 z-blis_sandybridge_4x4-small"
#
CONFIG_DESCRIPTION = ZGEMM kernel BLIS sandybridge_4x4, default block sizes
CONFIG_CPPFLAGS += -mavx -DULMBLAS_ZGEMM_UKERNEL_blis_sandybridge_4x4
CONFIG_REQUIRES  = avx

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
