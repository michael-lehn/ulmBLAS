#
#  Test configuration 09: small block sizes with a large register tile (MR =
#  32, NR = 16) and the reference micro kernel: the matrices of the tests are
#  often smaller than one tile, so that nearly all tiles are partial.
#
MC = 64
KC = 12
NC = 48

MR = 32
NR = 16

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR)

CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
