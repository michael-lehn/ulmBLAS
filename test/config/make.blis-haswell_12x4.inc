#
#  Test configuration blis-haswell_12x4: the BLIS-derived DGEMM micro kernel
#  haswell_12x4 (ulmblas/level3/ukernel/blis/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="blis-haswell_12x4 blis-haswell_12x4-small"
#
CONFIG_DESCRIPTION = BLIS kernel haswell_12x4, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_12x4
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
