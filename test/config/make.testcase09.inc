#
#  Test configuration 09: small block sizes with the experimental SME2 micro
#  kernel (MR = 32, NR = 16, which the kernel requires), so that the tests
#  (which use small matrices) cover multiple and partial blocks of all levels
#  of the GEMM algorithm with the optimized kernels.  The default block
#  sizes with the SME2 kernel are tested by configuration 04 if ARCHFLAGS
#  enable SME2 (e.g. ARCHFLAGS=-mcpu=apple-m4).  Skipped if the compiler or
#  the CPU do not support the kernel (see Makefile.config).
#
MC = 64
KC = 12
NC = 48

MR = 32
NR = 16

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), SME2 kernels

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
CONFIG_CPPFLAGS += -mcpu=apple-m4
CONFIG_REQUIRES  = sme

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
