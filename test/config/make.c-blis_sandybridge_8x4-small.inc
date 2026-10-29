#
#  Test configuration c-blis_sandybridge_8x4-small: the vendored CGEMM micro kernel
#  blis_sandybridge_8x4 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  MC and NC are multiples
#  of MR and NR of the CGEMM kernel, which fixes MR and NR for
#  std::complex<float>; MR and NR below are those of the other types (for
#  double: of the kernel ARCHFLAGS select).  See make.c-blis_sandybridge_8x4.inc.
#
MC = 24
KC = 11
NC = 24
MR = 4
NR = 4

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), CGEMM kernel BLIS sandybridge_8x4
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -DULMBLAS_CGEMM_UKERNEL_blis_sandybridge_8x4
CONFIG_REQUIRES  = avx

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
