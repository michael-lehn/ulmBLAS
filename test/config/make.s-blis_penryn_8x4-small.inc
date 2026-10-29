#
#  Test configuration s-blis_penryn_8x4-small: the vendored SGEMM micro kernel
#  blis_penryn_8x4 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  MC and NC are multiples
#  of MR and NR of the SGEMM kernel, which fixes MR and NR for float; MR
#  and NR below are those of the other types (for double: of the kernel
#  ARCHFLAGS select).  See make.s-blis_penryn_8x4.inc.
#
MC = 96
KC = 11
NC = 48
MR = 4
NR = 4

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), SGEMM kernel BLIS penryn_8x4
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -msse3 -DULMBLAS_SGEMM_UKERNEL_blis_penryn_8x4
CONFIG_REQUIRES  = sse3

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
