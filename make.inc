#
# Compilers
#

CXX := g++
CC  := gcc
FC  := gfortran

#
# Compiler and linker flags
#
CXXFLAGS   += -msse3 -mfpmath=sse
CXXFLAGS   += -std=c++11 -O3  -fomit-frame-pointer -Wall

FFLAGS     += -Wall -fimplicit-none -O3
LDLIBS     += -lm

#
# Enable precompiled headers
#
ULMBLAS_H     = $(TOPDIR)/ulmblas/ulmblas.h
ULMBLAS_PCH   = $(BUILD_DIR)/ulmblas.h.gch
ULMBLAS_D     = $(BUILD_DIR)/ulmblas.d
#INCLUDE_PCH   = -I $(BUILD_DIR) -include ulmblas.h
INCLUDE_PCH  = -include-pch $(ULMBLAS_PCH)
CXXFLAGS     += -Winvalid-pch


#
# Flags used for building the ATLAS benchmark suite
#
BENCH_CFLAGS := -c -DL2SIZE=4194304 -DAdd_ -DF77_INTEGER=int -DStringSunStyle \
                -DATL_SSE2 -DDREAL

#
#  Header file names
#
CLAPACK_HEADER_NAME   := clapack.h
ATLLAPACK_HEADER_NAME := atllapack.h

#
#  Recursive wildcard
#
rwildcard        = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))


# Flags used for linking Fortran with a C++ library.  We use the C++ compiler
# as linker font end.

FCLIBSEARCH := $(shell $(FC) -print-search-dirs  \
                      | grep libraries          \
                      | sed 's,libraries:.*=,,' \
                      | sed 's,:, ,g')
LDF77FLAGS += $(shell for path in $(FCLIBSEARCH); \
                      do                            \
                          if [ -d "$$path" ]; then    \
                              echo "-L $$path "; \
                          fi;                       \
                      done)
LDF77LIBS  += -lgfortran


