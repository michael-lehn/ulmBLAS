#
#  Test configuration blis-zen4_8x24: the BLIS-derived DGEMM micro kernel
#  zen4_8x24 (ulmblas/level3/ukernel/blis/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="blis-zen4_8x24 blis-zen4_8x24-small"
#
CONFIG_DESCRIPTION = BLIS kernel zen4_8x24, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_zen4_8x24
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
