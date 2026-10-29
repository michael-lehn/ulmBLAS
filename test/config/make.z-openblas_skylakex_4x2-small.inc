#
#  Test configuration z-openblas_skylakex_4x2-small: the vendored ZGEMM micro kernel
#  openblas_skylakex_4x2 with small block sizes (many partial blocks and panels; KC = 11
#  exercises the k loop remainders of the kernel).  MC and NC are multiples
#  of MR and NR of the ZGEMM kernel, which fixes MR and NR for
#  std::complex<double>; MR and NR below are those of the other types (for
#  double: of the kernel ARCHFLAGS select).  See make.z-openblas_skylakex_4x2.inc.
#
MC = 24
KC = 11
NC = 24
MR = 6
NR = 8

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), ZGEMM kernel OpenBLAS skylakex_4x2
CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -mavx -mfma -mavx512f -DULMBLAS_ZGEMM_UKERNEL_openblas_skylakex_4x2
CONFIG_REQUIRES  = avx fma avx512f

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
