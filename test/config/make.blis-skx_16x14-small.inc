#
#  Test configuration blis-skx_16x14-small: the BLIS-derived DGEMM micro kernel
#  skx_16x14 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  See make.blis-skx_16x14.inc.
#
MC = 112
KC = 11
NC = 28
MR = 16
NR = 14

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), BLIS kernel skx_16x14
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_skx_16x14
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
