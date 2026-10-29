#
#  Test configuration 06: default block sizes with the AVX/FMA micro kernel
#  (x86-64, -mavx -mfma).  Skipped if the compiler, the CPU or the operating
#  system do not support AVX and FMA (see Makefile.config).
#
CONFIG_DESCRIPTION = default block sizes, AVX/FMA kernels

CONFIG_CPPFLAGS += -mavx -mfma
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
