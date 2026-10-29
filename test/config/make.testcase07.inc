#
#  Test configuration 07: small block sizes with the AVX/FMA micro kernel
#  (MR = 6, NR = 8, which the kernel requires), so that the tests (which use
#  small matrices) cover multiple and partial blocks of all levels of the
#  GEMM algorithm with the optimized kernels.  Skipped if the compiler, the
#  CPU or the operating system do not support AVX and FMA (see
#  Makefile.config).
#
MC = 24
KC = 12
NC = 16

MR = 6
NR = 8

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), AVX/FMA kernels

CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
