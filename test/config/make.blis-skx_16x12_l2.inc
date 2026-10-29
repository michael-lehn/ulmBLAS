#
#  Test configuration blis-skx_16x12_l2: the BLIS-derived DGEMM micro kernel
#  skx_16x12_l2 (ulmblas/level3/ukernel/blis/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="blis-skx_16x12_l2 blis-skx_16x12_l2-small"
#
CONFIG_DESCRIPTION = BLIS kernel skx_16x12_l2, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_skx_16x12_l2
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
