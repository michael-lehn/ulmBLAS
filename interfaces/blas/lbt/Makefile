#
#  Shared library for libblastrampoline (LBT), the BLAS forwarding layer of
#  Julia: the Fortran 77 interface of ulmBLAS with 64-bit integers (ILP64)
#  and the symbol suffix _64_ (dgemm_64_), in one shared library
#
#      libulmblas_lbt.so      (Linux)
#      libulmblas_lbt.dylib   (macOS)
#
#  The objects of ../F77 are built with config/make.lbt.inc and -fPIC; the
#  Fortran functions (DDOT, ...) come from functions.c, so no Fortran
#  compiler is needed.  On Linux the C++ runtime is linked statically, so
#  the library does not depend on the libstdc++ of the compiler (Julia
#  brings its own, possibly older one).  ARCHFLAGS select the micro kernels
#  as for the other libraries; OUTDIR and BUILD_DIR allow several builds
#  side by side (e.g. one per instruction set).  In Julia:
#
#      using LinearAlgebra
#      BLAS.lbt_forward("/path/libulmblas_lbt.so"; clear=false)
#
TOPDIR = ../../..

include $(TOPDIR)/make.inc

OUTDIR     ?= $(TOPDIR)
BUILD_DIR  ?= build.lbt
LBT_INDEX   = int64_t
LBT_SUFFIX  = _64_

ifeq ($(shell uname -s),Darwin)
LBT_LIB      = $(OUTDIR)/libulmblas_lbt.dylib
LBT_LINK     = -dynamiclib -install_name @rpath/libulmblas_lbt.dylib \
               -Wl,-force_load,$(BUILD_DIR)/liblbtblas.a
else
LBT_LIB      = $(OUTDIR)/libulmblas_lbt.so
LBT_LINK     = -shared -Wl,-soname,libulmblas_lbt.so \
               -Wl,--whole-archive $(BUILD_DIR)/liblbtblas.a \
               -Wl,--no-whole-archive -static-libstdc++ -static-libgcc \
               -Wl,--no-undefined
endif

LBT_DEFS = -D'BLAS_HEADER=<lbtblas.h>' -D'F77BLAS(x)=x\#\#$(LBT_SUFFIX)'

.DELETE_ON_ERROR :
.PHONY : all clean FORCE

all : $(LBT_LIB)

#  The static library of the F77 interface (objects with -fPIC, without the
#  Fortran sources), by the makefile of ../F77
$(BUILD_DIR)/liblbtblas.a : FORCE
	$(MAKE) -C ../F77 -f Makefile.config CONFIG=lbt \
	        CONFIG_PATH=$(CURDIR)/ F77_SOURCES= \
	        OUTDIR=$(abspath $(BUILD_DIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/obj \
	        ULMBLAS_INDEX_TYPE=$(LBT_INDEX) CXXFLAGS="$(CXXFLAGS) -fPIC"

$(BUILD_DIR)/functions.o : functions.c $(BUILD_DIR)/liblbtblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -I$(BUILD_DIR) $(LBT_DEFS) \
	      -c -o $@ $<

$(LBT_LIB) : $(BUILD_DIR)/liblbtblas.a $(BUILD_DIR)/functions.o
	@mkdir -p $(OUTDIR)
	$(CXX) $(ARCHFLAGS) $(LDFLAGS) -o $@ $(BUILD_DIR)/functions.o $(LBT_LINK) -lm

clean :
	$(RM) -r $(BUILD_DIR) $(LBT_LIB)
