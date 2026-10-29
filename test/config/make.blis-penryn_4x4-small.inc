#
#  Test configuration blis-penryn_4x4-small: the BLIS-derived DGEMM micro kernel
#  penryn_4x4 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  See make.blis-penryn_4x4.inc.
#
MC = 8
KC = 11
NC = 8
MR = 4
NR = 4

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), BLIS kernel penryn_4x4
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -msse3 -DULMBLAS_DGEMM_UKERNEL_blis_penryn_4x4
CONFIG_REQUIRES  = sse3

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
