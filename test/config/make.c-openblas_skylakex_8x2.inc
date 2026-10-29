#
#  Test configuration c-openblas_skylakex_8x2: the vendored CGEMM micro kernel openblas_skylakex_8x2
#  (ulmblas/level3/ukernel/blis/select_c.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="c-openblas_skylakex_8x2 c-openblas_skylakex_8x2-small"
#
CONFIG_DESCRIPTION = CGEMM kernel OpenBLAS skylakex_8x2, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
