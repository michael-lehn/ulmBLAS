#
#  Test configuration c-blis_haswell_3x8-small: the vendored CGEMM micro kernel
#  blis_haswell_3x8 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  MC and NC are multiples
#  of MR and NR of the CGEMM kernel, which fixes MR and NR for
#  std::complex<float>; MR and NR below are those of the other types (for
#  double: of the kernel ARCHFLAGS select).  See make.c-blis_haswell_3x8.inc.
#
MC = 24
KC = 11
NC = 24
MR = 6
NR = 8

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), CGEMM kernel BLIS haswell_3x8
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
