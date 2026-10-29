#
#  Test configuration z-openblas_skylakex_4x2: the vendored ZGEMM micro kernel openblas_skylakex_4x2
#  (ulmblas/level3/ukernel/blis/select_z.h) with its default block sizes.
#  Not part of the default configurations (testcase*); run with
#
#      make check CONFIGS="z-openblas_skylakex_4x2 z-openblas_skylakex_4x2-small"
#
CONFIG_DESCRIPTION = ZGEMM kernel OpenBLAS skylakex_4x2, default block sizes
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_ZGEMM_UKERNEL_openblas_skylakex_4x2
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
