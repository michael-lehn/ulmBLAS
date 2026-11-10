#
#  Shared library with the standard BLAS interfaces, for programs and
#  packages that link a system BLAS (NumPy, SciPy, R, FlexiBLAS, ...):
#
#      libulmblas.so      (Linux)
#      libulmblas.dylib   (macOS)
#
#  It contains the Fortran 77 interface with 32-bit integers (LP64) and the
#  usual names (dgemm_, ddot_, xerbla_, lsame_, ...) and the C interface (CBLAS:
#  cblas_dgemm, ..., cblas_xerbla, and the ulmblas_ functions including
#  GEMM with arbitrary strides).  The objects of ../F77 are built with
#  config/make.shared.inc and -fPIC; the Fortran functions (DDOT, ...)
#  come from ../lbt/functions.c, so no Fortran compiler is needed.  The
#  return values of the complex functions (ZDOTC, ...) follow gfortran.
#  On Linux the C++ runtime is linked statically.  ARCHFLAGS select the
#  micro kernels as for the other libraries; OUTDIR and BUILD_DIR allow
#  several builds side by side.
#
#  The headers of the two interfaces are generated in BUILD_DIR: cblas.h
#  (and ulmblas_strided.h) and sharedblas.h (F77 prototypes).
#
TOPDIR = ../../..

include $(TOPDIR)/make.inc

OUTDIR     ?= $(TOPDIR)
BUILD_DIR  ?= build.shared

SHARED_INDEX  = int
SHARED_SUFFIX = _

EXPORTS = $(BUILD_DIR)/exports.txt

ifeq ($(shell uname -s),Darwin)
SHARED_LIB   = $(OUTDIR)/libulmblas.dylib
SHARED_LINK  = -dynamiclib -install_name @rpath/libulmblas.dylib \
               -Wl,-force_load,$(BUILD_DIR)/libsharedblas.a \
               -Wl,-force_load,$(BUILD_DIR)/libcblas.a \
               -Wl,-exported_symbols_list,$(EXPORTS)
SYMBOL_PREFIX = _
else
SHARED_LIB   = $(OUTDIR)/libulmblas.so
SHARED_LINK  = -shared -Wl,-soname,libulmblas.so \
               -Wl,--whole-archive $(BUILD_DIR)/libsharedblas.a \
               $(BUILD_DIR)/libcblas.a -Wl,--no-whole-archive \
               -static-libstdc++ -static-libgcc -Wl,--no-undefined \
               -Wl,--version-script,$(EXPORTS)
SYMBOL_PREFIX =
endif

F77_DEFS = -D'BLAS_HEADER=<sharedblas.h>' -D'F77BLAS(x)=x\#\#$(SHARED_SUFFIX)'

.DELETE_ON_ERROR :
.PHONY : all clean FORCE

all : $(SHARED_LIB)

#  The static library of the F77 interface (objects with -fPIC, without the
#  Fortran sources), by the makefile of ../F77; it also contains the
#  assembly kernels of the configuration (ULMBLAS_ASM_OBJECTS), if any
$(BUILD_DIR)/libsharedblas.a : FORCE
	$(MAKE) -C ../F77 -f Makefile.config CONFIG=shared \
	        CONFIG_PATH=$(CURDIR)/ F77_SOURCES= \
	        OUTDIR=$(abspath $(BUILD_DIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/obj-f77 \
	        ULMBLAS_INDEX_TYPE=$(SHARED_INDEX) \
	        CXXFLAGS="$(CXXFLAGS) -fPIC -DULMBLAS_REPLACEABLE_XERBLA"

#  The static library of the C interface, by the makefile of ../C, without
#  the assembly kernels (they are in the F77 library)
$(BUILD_DIR)/libcblas.a : FORCE
	$(MAKE) -C ../C -f Makefile.config CONFIG=cblas \
	        OUTDIR=$(abspath $(BUILD_DIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/obj-c \
	        ULMBLAS_ASM_OBJECTS= \
	        ULMBLAS_INDEX_TYPE=$(SHARED_INDEX) \
	        CXXFLAGS="$(CXXFLAGS) -fPIC -DULMBLAS_REPLACEABLE_XERBLA"

$(BUILD_DIR)/functions.o : ../lbt/functions.c $(BUILD_DIR)/libsharedblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -I$(BUILD_DIR) $(F77_DEFS) \
	      -c -o $@ $<

$(BUILD_DIR)/lsame.o : lsame.c $(BUILD_DIR)/libsharedblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -c -o $@ $<

SHARED_OBJECTS = $(BUILD_DIR)/functions.o $(BUILD_DIR)/lsame.o

#  Only the interfaces are exported: the Fortran 77 names (lowercase with
#  one trailing underscore, without the helpers *_sub_), cblas_*,
#  ulmblas_* and the two globals of the reference CBLAS.  Not exported: the
#  C++ templates of ulmBLAS (they could clash with another library that
#  contains them).  The list is taken from the objects (nm), as a list of
#  names (macOS) or a version script (Linux).
EXPORT_PATTERN = '^$(SYMBOL_PREFIX)([a-z0-9]+_|cblas_[a-z0-9_]+|ulmblas_[a-z0-9_]+|RowMajorStrg|CBLAS_CallFromC)$$'

$(EXPORTS) : $(BUILD_DIR)/libsharedblas.a $(BUILD_DIR)/libcblas.a \
             $(SHARED_OBJECTS)
	nm -g $^ | awk 'NF==3 && $$2 ~ /^[TDBRSVW]$$/ {print $$3}' \
	    | grep -E $(EXPORT_PATTERN) | grep -v '_sub_$$' | sort -u > $@.tmp
	@if [ "$(SYMBOL_PREFIX)" = "" ]; then \
	    { echo '{ global:'; sed 's/$$/;/' $@.tmp; echo 'local: *; };'; } > $@; \
	    rm -f $@.tmp; \
	else \
	    mv $@.tmp $@; \
	fi

#  No C++ exception may pass through the interface (it is called from C
#  and Fortran): the objects of ulmBLAS must not refer to functions that
#  throw (operator new, std::__throw_*, __cxa_throw, ...).  Checked before
#  linking.
THROWING = '_Zn[wa][jmy](St11align_val_t)?$$|__cxa_throw$$|__cxa_rethrow$$|__cxa_allocate_exception$$|_ZSt[0-9]+__throw_'

$(SHARED_LIB) : $(BUILD_DIR)/libsharedblas.a $(BUILD_DIR)/libcblas.a \
                $(SHARED_OBJECTS) $(EXPORTS)
	@if nm -u $(filter-out $(EXPORTS),$^) | grep -E $(THROWING); then \
	    echo "error: objects refer to functions that throw (above)"; \
	    exit 1; \
	fi
	@mkdir -p $(OUTDIR)
	$(CXX) $(ARCHFLAGS) $(LDFLAGS) -o $@ $(SHARED_OBJECTS) $(SHARED_LINK) \
	       $(OMPLIBS) -lm

clean :
	$(RM) -r $(BUILD_DIR) $(SHARED_LIB)
