#
#  Test configuration c-blis_haswell_3x8: the vendored CGEMM micro kernel blis_haswell_3x8
#  (ulmblas/level3/ukernel/blis/select_c.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="c-blis_haswell_3x8 c-blis_haswell_3x8-small"
#
CONFIG_DESCRIPTION = CGEMM kernel BLIS haswell_3x8, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
