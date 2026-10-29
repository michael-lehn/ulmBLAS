#
#  Test configuration blis-sandybridge_int_8x4: the BLIS-derived DGEMM micro kernel
#  sandybridge_int_8x4 (ulmblas/level3/ukernel/blis/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="blis-sandybridge_int_8x4 blis-sandybridge_int_8x4-small"
#
CONFIG_DESCRIPTION = BLIS kernel sandybridge_int_8x4, default block sizes
CONFIG_CPPFLAGS += -mavx -DULMBLAS_DGEMM_UKERNEL_blis_sandybridge_int_8x4
CONFIG_REQUIRES  = avx

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
