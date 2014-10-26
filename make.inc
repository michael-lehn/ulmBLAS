#
# Compilers
#

CXX = g++
CC  = gcc
FC  = gfortran

#
# Compiler and linker flags
#
CXXFLAGS += -msse3 -mfpmath=sse
CXXFLAGS += -std=c++11 -O3  -fomit-frame-pointer -Wall

FFLAGS   += -fimplicit-none -O3
LDLIBS   += -lm

#
# Flags used for building the ATLAS benchmark suite
#
BENCH_CFLAGS = -c -DL2SIZE=4194304 -DAdd_ -DF77_INTEGER=int -DStringSunStyle \
               -DATL_SSE2 -DDREAL
