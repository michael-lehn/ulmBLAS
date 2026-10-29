#
#  Test configuration blis-zen5_8x24-small: the BLIS-derived DGEMM micro kernel
#  zen5_8x24 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  See make.blis-zen5_8x24.inc.
#
MC = 24
KC = 11
NC = 48
MR = 8
NR = 24

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), BLIS kernel zen5_8x24
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_zen5_8x24
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
