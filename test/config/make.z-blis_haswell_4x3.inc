#
#  Test configuration z-blis_haswell_4x3: the vendored ZGEMM micro kernel blis_haswell_4x3
#  (ulmblas/level3/ukernel/blis/select_z.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="z-blis_haswell_4x3 z-blis_haswell_4x3-small"
#
CONFIG_DESCRIPTION = ZGEMM kernel BLIS haswell_4x3, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_haswell_4x3
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
