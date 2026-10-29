#
#  Test configuration 04: the default block sizes and kernels of ulmBLAS
#  (on x86 with SSE3 the optimized SSE kernels), as used by the libraries
#  built by 'make'.
#
CONFIG_DESCRIPTION = default block sizes and kernels

include $(dir $(lastword $(MAKEFILE_LIST)))make.names.inc
