#
#  Test configuration openblas-haswell_4x8: the OpenBLAS-derived DGEMM micro kernel
#  haswell_4x8 (ulmblas/level3/ukernel/openblas/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="openblas-haswell_4x8 openblas-haswell_4x8-small openblas-haswell_4x8-kc600"
#
CONFIG_DESCRIPTION = OpenBLAS kernel haswell_4x8, default block sizes
CONFIG_CPPFLAGS += -mavx -mavx2 -mfma -DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8
CONFIG_REQUIRES  = avx avx2 fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
