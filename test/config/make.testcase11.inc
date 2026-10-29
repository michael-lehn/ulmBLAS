#
#  Test configuration 11: the small block sizes of configuration 01 with a
#  64-bit index type in the interfaces given as two words
#  (ULMBLAS_INDEX_TYPE = long long); Fortran 77 and the Netlib test
#  programs as in configuration 10.
#
MC = 8
KC = 12
NC = 12

MR = 4
NR = 4

ULMBLAS_INDEX_TYPE = long long

CONFIG_DESCRIPTION = MC=$(MC) KC=$(KC) NC=$(NC) MR=$(MR) NR=$(NR), ULMBLAS_INDEX_TYPE=$(ULMBLAS_INDEX_TYPE)

CONFIG_CPPFLAGS += -DUSE_TESTPARAM
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MC=$(MC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_KC=$(KC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NC=$(NC)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_MR=$(MR)
CONFIG_CPPFLAGS += -DUSE_TESTPARAM_NR=$(NR)

FFLAGS += -fdefault-integer-8

CONFIG_NETLIB_C_SKIP = the Netlib CBLAS test programs use int arguments (ULMBLAS_INDEX_TYPE=$(ULMBLAS_INDEX_TYPE))

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
