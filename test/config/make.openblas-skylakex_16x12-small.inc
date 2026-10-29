#
#  Test configuration openblas-skylakex_16x12-small: the OpenBLAS-derived DGEMM micro kernel
#  skylakex_16x12 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  See make.openblas-skylakex_16x12.inc.
#
MC = 48
KC = 11
NC = 24
MR = 16
NR = 12

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), OpenBLAS kernel skylakex_16x12
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
