#
#  Test configuration 10: default block sizes and kernels with a 64-bit
#  index type in the interfaces (ULMBLAS_INDEX_TYPE = int64_t).  The
#  Fortran 77 code (the wrappers of the interface and the Netlib test
#  programs) is compiled with 64-bit INTEGER; the Netlib CBLAS test
#  programs pass int arguments and are skipped.
#
ULMBLAS_INDEX_TYPE = int64_t

CONFIG_DESCRIPTION = default block sizes and kernels, ULMBLAS_INDEX_TYPE=$(ULMBLAS_INDEX_TYPE)

FFLAGS += -fdefault-integer-8

CONFIG_NETLIB_C_SKIP = the Netlib CBLAS test programs use int arguments (ULMBLAS_INDEX_TYPE=$(ULMBLAS_INDEX_TYPE))

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
