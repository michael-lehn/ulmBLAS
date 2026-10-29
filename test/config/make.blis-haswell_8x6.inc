#
#  Test configuration blis-haswell_8x6: the BLIS-derived DGEMM micro kernel
#  haswell_8x6 (ulmblas/level3/ukernel/blis/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="blis-haswell_8x6 blis-haswell_8x6-small"
#
CONFIG_DESCRIPTION = BLIS kernel haswell_8x6, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_8x6
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
