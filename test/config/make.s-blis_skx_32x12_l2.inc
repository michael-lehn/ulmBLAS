#
#  Test configuration s-blis_skx_32x12_l2: the vendored SGEMM micro kernel blis_skx_32x12_l2
#  (ulmblas/level3/ukernel/blis/select_s.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="s-blis_skx_32x12_l2 s-blis_skx_32x12_l2-small"
#
CONFIG_DESCRIPTION = SGEMM kernel BLIS skx_32x12_l2, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_SGEMM_UKERNEL_blis_skx_32x12_l2
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
