#
#  Test configuration 08: small block sizes with the NEON micro kernel
#  (AArch64, MR = 6, NR = 8, which the kernel requires), so that the tests
#  (which use small matrices) cover multiple and partial blocks of all levels
#  of the GEMM algorithm with the optimized kernels.  The default block sizes
#  with the NEON kernel are tested by configuration 04 on AArch64.  Skipped
#  on other architectures (see Makefile.config).  -DULMBLAS_NO_SME keeps the
#  NEON kernel if ARCHFLAGS enable SME2 (e.g. -mcpu=apple-m4).
#
MC = 24
KC = 12
NC = 16

MR = 6
NR = 8

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), NEON kernels

CONFIG_CPPFLAGS += -DUSE_TESTPARAM
#  without the direct path for small problems (gemmdirect.tcc), so that the
#  small test problems use the blocked algorithm
CONFIG_CPPFLAGS += -DULMBLAS_GEMM_DIRECT_MAX_MNK=0
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_CPPFLAGS += -DULMBLAS_NO_SME
CONFIG_REQUIRES  = neon

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
