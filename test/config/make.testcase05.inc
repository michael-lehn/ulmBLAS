#
#  Test configuration 05: small block sizes with MR = 6 and NR = 8 (the block
#  size of the AVX micro kernel for double), which do not divide each other,
#  with the reference micro kernel.  Like configurations 01 to 03 the tests
#  (which use small matrices) cover multiple and partial blocks of all levels
#  of the GEMM algorithm.  Libraries, headers and tests of a configuration are
#  built in test/build/<configuration>/ (see test/Makefile.config).
#
MC = 24
KC = 12
NC = 16

MR = 6
NR = 8

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR)

CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
