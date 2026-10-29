#
#  Test configuration s-blis_aocl_haswell_6x16-small: the vendored SGEMM micro kernel
#  blis_aocl_haswell_6x16 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  MC and NC are multiples
#  of MR and NR of the SGEMM kernel, which fixes MR and NR for float; MR
#  and NR below are those of the other types (for double: of the kernel
#  ARCHFLAGS select).  See make.s-blis_aocl_haswell_6x16.inc.
#
MC = 96
KC = 11
NC = 48
MR = 6
NR = 8

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), SGEMM kernel AOCL haswell_6x16
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_aocl_haswell_6x16
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
