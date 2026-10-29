#
#  Test configuration z-blis_haswell_4x3-small: the vendored ZGEMM micro kernel
#  blis_haswell_4x3 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  MC and NC are multiples
#  of MR and NR of the ZGEMM kernel, which fixes MR and NR for
#  std::complex<double>; MR and NR below are those of the other types (for
#  double: of the kernel ARCHFLAGS select).  See make.z-blis_haswell_4x3.inc.
#
MC = 24
KC = 11
NC = 24
MR = 6
NR = 8

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), ZGEMM kernel BLIS haswell_4x3
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_haswell_4x3
CONFIG_REQUIRES  = avx fma

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
