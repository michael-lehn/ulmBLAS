#
#  Test configuration blis-zen4_32x6-small: the BLIS-derived DGEMM micro kernel
#  zen4_32x6 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  See make.blis-zen4_32x6.inc.
#
MC = 96
KC = 11
NC = 12
MR = 32
NR = 6

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), BLIS kernel zen4_32x6
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_zen4_32x6
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
