#
#  Test configuration 06: default block sizes and kernels in a build with
#  -mavx -mfma: the compiler generates VEX-encoded code around the inline
#  assembly of the SSE kernels (as with ARCHFLAGS=-march=native).  Skipped if
#  the compiler, the CPU or the operating system do not support AVX and FMA
#  (see Makefile.config).
#
CONFIG_DESCRIPTION = default block sizes and kernels, -mavx -mfma

CONFIG_CPPFLAGS += -mavx -mfma
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
