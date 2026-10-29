#
#  Test configuration z-blis_zen4_4x12: the vendored ZGEMM micro kernel blis_zen4_4x12
#  (ulmblas/level3/ukernel/blis/select_z.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="z-blis_zen4_4x12 z-blis_zen4_4x12-small"
#
CONFIG_DESCRIPTION = ZGEMM kernel AOCL zen4_4x12, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -mavx512dq -DULMBLAS_ZGEMM_UKERNEL_blis_zen4_4x12
CONFIG_REQUIRES  = avx fma avx512f avx512dq

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
