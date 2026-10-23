#
#  Test configuration 01: small block sizes, so that the tests (which use
#  small matrices) cover multiple and partial blocks of all levels of the
#  GEMM algorithm.  Libraries, headers and tests of a configuration are
#  built in test/build/<configuration>/ (see test/Makefile.config).
#
MC = 8
KC = 12
NC = 12

MR = 4
NR = 4

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR)

CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
