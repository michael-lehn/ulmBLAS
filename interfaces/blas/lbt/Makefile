#
#  Shared library for libblastrampoline (LBT), the BLAS forwarding layer of
#  Julia: the Fortran 77 interface of ulmBLAS with 64-bit integers (ILP64)
#  and the symbol suffix _64_ (dgemm_64_), in one shared library
#
#      libulmblas_lbt.so      (Linux)
#      libulmblas_lbt.dylib   (macOS)
#      libulmblas_lbt.dll     (Windows, MinGW-w64)
#
#  The objects of ../F77 are built with config/make.lbt.inc and -fPIC; the
#  Fortran functions (DDOT, ...) come from functions.c, so no Fortran
#  compiler is needed.  The library also contains GEMM with arbitrary
#  strides (ulmblas_?gemm_strided, ../C/ulmblas_strided.h);
#  libblastrampoline does not know these names and ignores them.  On Linux
#  the C++ runtime is linked statically, so the library does not depend on
#  the libstdc++ of the compiler (Julia
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

#  LAPACK=1: the library also contains ?getrf_64_ and ?potrf_64_ (../../
#  lapack/F77).  Off by default: libblastrampoline would then forward them
#  to ulmBLAS instead of OpenBLAS, whose versions are multithreaded; ulmBLAS
#  factors in parallel only if it is built with OpenMP.
LAPACK     ?= 0
comma      := ,
LBT_LAPACK  = $(if $(filter 1,$(LAPACK)),$(BUILD_DIR)/liblbtlapack.a)
LBT_INDEX   = int64_t
LBT_SUFFIX  = _64_

#  Windows (MinGW-w64): all global symbols of the objects are exported (no
#  symbol is marked dllexport), the C++ runtime and winpthread are linked
#  statically, so the DLL only depends on DLLs of the system.
ifneq (,$(findstring mingw,$(shell $(CXX) -dumpmachine)))
LBT_LIB      = $(OUTDIR)/libulmblas_lbt.dll
LBT_LINK     = -shared -Wl,--whole-archive $(BUILD_DIR)/liblbtblas.a \
               -Wl,--no-whole-archive -static -static-libstdc++ \
               -static-libgcc -Wl,--no-undefined
else ifeq ($(shell uname -s),Darwin)
LBT_LIB      = $(OUTDIR)/libulmblas_lbt.dylib
LBT_LINK     = -dynamiclib -install_name @rpath/libulmblas_lbt.dylib \
               -Wl,-force_load,$(BUILD_DIR)/liblbtblas.a \
               $(if $(LBT_LAPACK),-Wl$(comma)-force_load$(comma)$(LBT_LAPACK))
else
LBT_LIB      = $(OUTDIR)/libulmblas_lbt.so
LBT_LINK     = -shared -Wl,-soname,libulmblas_lbt.so \
               -Wl,--whole-archive $(BUILD_DIR)/liblbtblas.a \
               $(LBT_LAPACK) -Wl,--no-whole-archive -static-libstdc++ -static-libgcc \
               -Wl,--no-undefined
endif

LBT_DEFS = -D'BLAS_HEADER=<lbtblas.h>' -D'F77BLAS(x)=x\#\#$(LBT_SUFFIX)'
STRIDED_DEFS = -D'ULMBLAS(x)=ulmblas_\#\#x'

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

#  The Fortran 77 interface of the LAPACK functions (?getrf_64_,
#  ?potrf_64_), by the makefile of ../../lapack/F77
$(BUILD_DIR)/liblbtlapack.a : $(BUILD_DIR)/liblbtblas.a FORCE
	$(MAKE) -C ../../lapack/F77 -f Makefile.config CONFIG=lbt \
	        CONFIG_PATH=$(CURDIR)/ \
	        OUTDIR=$(abspath $(BUILD_DIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/obj-lapack \
	        ULMBLAS_INDEX_TYPE=$(LBT_INDEX) CXXFLAGS="$(CXXFLAGS) -fPIC"

$(BUILD_DIR)/functions.o : functions.c $(BUILD_DIR)/liblbtblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -I$(BUILD_DIR) $(LBT_DEFS) \
	      -c -o $@ $<

#  GEMM with arbitrary strides, from the C interface
$(BUILD_DIR)/gemm_strided.o : ../C/gemm_strided.cc ../C/ulmblas_strided.h \
                              $(BUILD_DIR)/liblbtblas.a
	$(CXX) -I$(TOPDIR) $(STRIDED_DEFS) $(ULMBLAS_CXXFLAGS) $(CPPFLAGS) \
	       $(CXXFLAGS) -fPIC -c -o $@ $<

LBT_OBJECTS = $(BUILD_DIR)/functions.o $(BUILD_DIR)/gemm_strided.o

#  No C++ exception may pass through the interface (Julia calls it from C):
#  the objects of ulmBLAS must not refer to functions that throw (operator
#  new, std::__throw_*, __cxa_throw, ...).  Checked before linking.
THROWING = '_Zn[wa][jmy](St11align_val_t)?$$|__cxa_throw$$|__cxa_rethrow$$|__cxa_allocate_exception$$|_ZSt[0-9]+__throw_'

$(LBT_LIB) : $(BUILD_DIR)/liblbtblas.a $(LBT_LAPACK) $(LBT_OBJECTS)
	@if nm -u $^ | grep -E $(THROWING); then \
	    echo "error: objects refer to functions that throw (above)"; \
	    exit 1; \
	fi
	@mkdir -p $(OUTDIR)
	$(CXX) $(ARCHFLAGS) $(LDFLAGS) -o $@ $(LBT_OBJECTS) $(LBT_LINK) $(OMPLIBS) -lm

clean :
	$(RM) -r $(BUILD_DIR) $(LBT_LIB)
