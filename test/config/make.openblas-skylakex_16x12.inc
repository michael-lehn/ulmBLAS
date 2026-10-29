#
#  Test configuration openblas-skylakex_16x12: the OpenBLAS-derived DGEMM micro kernel
#  skylakex_16x12 (ulmblas/level3/ukernel/openblas/) with its default block sizes.  Not
#  part of the default configurations (testcase*); run with
#
#      make check CONFIGS="openblas-skylakex_16x12 openblas-skylakex_16x12-small"
#
CONFIG_DESCRIPTION = OpenBLAS kernel skylakex_16x12, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
