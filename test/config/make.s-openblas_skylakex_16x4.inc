#
#  Test configuration s-openblas_skylakex_16x4: the vendored SGEMM micro kernel openblas_skylakex_16x4
#  (ulmblas/level3/ukernel/blis/select_s.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="s-openblas_skylakex_16x4 s-openblas_skylakex_16x4-small"
#
CONFIG_DESCRIPTION = SGEMM kernel OpenBLAS skylakex_16x4, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
