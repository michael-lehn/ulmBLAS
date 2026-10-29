#
#  Test configuration blis-zen4_32x6: the BLIS-derived DGEMM micro kernel
#  zen4_32x6 (ulmblas/level3/ukernel/blis/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="blis-zen4_32x6 blis-zen4_32x6-small"
#
CONFIG_DESCRIPTION = BLIS kernel zen4_32x6, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_zen4_32x6
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
