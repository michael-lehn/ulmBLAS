#
#  Test configuration blis-haswell_8x6-small: the BLIS-derived DGEMM micro kernel
#  haswell_8x6 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  See make.blis-haswell_8x6.inc.
#
MC = 24
KC = 11
NC = 12
MR = 8
NR = 6

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), BLIS kernel haswell_8x6
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_8x6
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
