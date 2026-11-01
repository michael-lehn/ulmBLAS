#
#  Test configuration 07: small block sizes with the SSE micro kernels (MR =
#  NR = 4, which the SSE TRSM kernels require), so that the tests (which use
#  small matrices) cover multiple and partial blocks of all levels of the
#  GEMM algorithm with the optimized kernels.  Skipped on other architectures
#  (see Makefile.config).
#
MC = 8
KC = 12
NC = 12

MR = 4
NR = 4

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), SSE kernels

CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KEEP_KERNELS
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)
CONFIG_REQUIRES  = sse3

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
