#
#  Shared library with the standard BLAS interfaces, for programs and
#  packages that link a system BLAS (NumPy, SciPy, R, FlexiBLAS, ...):
#
#      libulmblas.so      (Linux)
#      libulmblas.dylib   (macOS)
#      libulmblas.dll     (Windows, MinGW-w64; import library libulmblas.dll.a)
#
#  SHARED_VERSION=0: with the version in the name, libulmblas.so.0 (soname)
#  and libulmblas.0.dylib (install name), and a link without it; the
#  default is no version (the backends of ../dispatch).  The platform is
#  that of the compiler ($(CXX) -dumpmachine), so cross compilers work
#  (e.g. CXX=x86_64-w64-mingw32-g++ CC=x86_64-w64-mingw32-gcc
#  NM=x86_64-w64-mingw32-nm AR=x86_64-w64-mingw32-ar
#  RANLIB=x86_64-w64-mingw32-ranlib).
#
#  It contains the Fortran 77 interface with 32-bit integers (LP64) and the
#  usual names (dgemm_, ddot_, xerbla_, lsame_, ...) and the C interface (CBLAS:
#  cblas_dgemm, ..., cblas_xerbla, and the ulmblas_ functions including
#  GEMM with arbitrary strides), and the LAPACK functions of ulmBLAS
#  (LAPACK_EXPORTS: ?getrf, ?potrf, ...; the libulmlapack of
#  ../../lapack/reference uses them instead of its own routines).  The
#  objects of ../F77 are built with config/make.shared.inc and -fPIC; the
#  Fortran functions (DDOT, ...) come from ../lbt/functions.c, so no
#  Fortran compiler is needed.  The return values of the complex functions
#  (ZDOTC, ...) follow gfortran.  On Linux the C++ runtime is linked
#  statically if libstdc++.a exists.  ARCHFLAGS select the micro kernels as
#  for the other libraries; OUTDIR and BUILD_DIR allow several builds side
#  by side.
#
#  ILP64=1 (default): the library also contains both interfaces with
#  64-bit integers and the names of the reference BLAS and CBLAS built
#  with BUILD_INDEX64_EXT_API (Arch, openSUSE): dgemm_64_, ...,
#  xerbla_64_, xerbla_array_64_, lsame_64_ and cblas_dgemm_64, ...,
#  cblas_xerbla_64.  The LAPACK of these distributions (liblapack.so.3)
#  calls them.  The same sources, built a second time with
#  ULMBLAS_INDEX_TYPE=int64_t (config/make.shared64.inc,
#  config/make.cblas64.inc); the functions behind the interfaces are the
#  same (instantiated with std::ptrdiff_t).  ILP64=0: LP64 only.
#
#  The headers of the two interfaces are generated in BUILD_DIR: cblas.h
#  (and ulmblas_strided.h) and sharedblas.h (F77 prototypes); those of the
#  ILP64 interfaces in BUILD_DIR/ilp64 (cblas64.h, shared64blas.h).
#
#  INTERFACE64=1: libulmblas64 instead, the interfaces with 64-bit integers
#  under the usual names (dgemm_, cblas_dgemm, lsame_ with a 64-bit
#  LOGICAL, xerbla_, ...), as Debian's libblas64: the library for programs
#  compiled with 64-bit INTEGER (gfortran -fdefault-integer-8) and the
#  backend of FlexiBLAS with 64-bit integers (libflexiblas64), which looks
#  up only these names.  Without the LP64 interfaces and without the names
#  with _64_ (ILP64=0); its headers (cblas.h, sharedblas.h) have the 64-bit
#  index type.  A program must not load libulmblas and libulmblas64
#  together: the names are the same, the integers are not.  On macOS and
#  Linux on x86-64, ../dispatch builds libulmblas64 as a front of the
#  backends of libulmblas.
#
TOPDIR = ../../..

include $(TOPDIR)/make.inc

INTERFACE64 ?= 0

ifeq ($(INTERFACE64),1)
OUTDIR     ?= $(TOPDIR)
BUILD_DIR  ?= build.shared64
SHARED_NAME ?= ulmblas64
SHARED_INDEX  = int64_t
SHARED_LOGICAL = int64_t
override ILP64 = 0
else
OUTDIR     ?= $(TOPDIR)
BUILD_DIR  ?= build.shared
#  SHARED_NAME: the library is lib$(SHARED_NAME).so / .dylib (the backends
#  of the dispatching library on macOS are built with other names, see
#  ../dispatch)
SHARED_NAME ?= ulmblas
SHARED_INDEX  = int
SHARED_LOGICAL = int
endif

SHARED_SUFFIX = _

SHARED_VERSION ?=
NM ?= nm

TARGET_MACHINE := $(shell $(CXX) -dumpmachine)
COMMA := ,

ifneq (,$(findstring mingw,$(TARGET_MACHINE)))
#  Windows: only the names of the .def file are exported; the C++ runtime
#  and winpthread are linked statically (the DLL depends only on DLLs of
#  the system)
EXPORTS      = $(BUILD_DIR)/exports.def
SHARED_LIB   = $(OUTDIR)/lib$(SHARED_NAME).dll
SHARED_LINK  = -shared -Wl,--out-implib,$(OUTDIR)/lib$(SHARED_NAME).dll.a \
               $(EXPORTS) \
               -Wl,--whole-archive $(BUILD_DIR)/libsharedblas.a \
               $(BUILD_DIR)/libcblas.a $(LAPACK_LIB) $(ILP64_LIBS) \
               -Wl,--no-whole-archive \
               -static -static-libstdc++ -static-libgcc -Wl,--no-undefined
SHARED_ALIAS =
SYMBOL_PREFIX =
else ifneq (,$(findstring apple-darwin,$(TARGET_MACHINE)))
EXPORTS      = $(BUILD_DIR)/exports.txt
SHARED_FILE  = lib$(SHARED_NAME)$(if $(SHARED_VERSION),.$(SHARED_VERSION)).dylib
SHARED_LIB   = $(OUTDIR)/$(SHARED_FILE)
SHARED_LINK  = -dynamiclib -install_name @rpath/$(SHARED_FILE) \
               -Wl,-force_load,$(BUILD_DIR)/libsharedblas.a \
               -Wl,-force_load,$(BUILD_DIR)/libcblas.a \
               -Wl,-force_load,$(LAPACK_LIB) \
               $(foreach l,$(ILP64_LIBS),-Wl$(COMMA)-force_load$(COMMA)$(l)) \
               -Wl,-exported_symbols_list,$(EXPORTS)
SHARED_ALIAS = $(if $(SHARED_VERSION),$(OUTDIR)/lib$(SHARED_NAME).dylib)
SYMBOL_PREFIX = _
else
EXPORTS      = $(BUILD_DIR)/exports.txt
SHARED_FILE  = lib$(SHARED_NAME).so$(if $(SHARED_VERSION),.$(SHARED_VERSION))
SHARED_LIB   = $(OUTDIR)/$(SHARED_FILE)
SHARED_LINK  = -shared -Wl,-soname,$(SHARED_FILE) \
               -Wl,--whole-archive $(BUILD_DIR)/libsharedblas.a \
               $(BUILD_DIR)/libcblas.a $(LAPACK_LIB) $(ILP64_LIBS) \
               -Wl,--no-whole-archive \
               $(STATIC_LIBSTDCXX) -static-libgcc -pthread -Wl,--no-undefined \
               -Wl,--version-script,$(EXPORTS)
#  libstdc++.a is missing on some systems (RHEL: package libstdc++-static);
#  then the C++ runtime is linked dynamically.  -pthread: the thread pool
#  (libpthread is a library of its own before glibc 2.34)
STATIC_LIBSTDCXX := $(if $(filter /%,$(shell $(CXX) \
                    -print-file-name=libstdc++.a)),-static-libstdc++)
SHARED_ALIAS = $(if $(SHARED_VERSION),$(OUTDIR)/lib$(SHARED_NAME).so)
SYMBOL_PREFIX =
endif

F77_DEFS = -D'BLAS_HEADER=<sharedblas.h>' -D'F77BLAS(x)=x\#\#$(SHARED_SUFFIX)'

ILP64        ?= 1
ILP64_INDEX   = int64_t
ILP64_SUFFIX  = _64_
ILP64_DIR     = $(BUILD_DIR)/ilp64
F77_DEFS_64   = -D'BLAS_HEADER=<shared64blas.h>' \
                -D'F77BLAS(x)=x\#\#$(ILP64_SUFFIX)'

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

#  Our LAPACK functions (../../lapack/F77: ?getrf, ?potrf, ...), with the
#  configuration of the Fortran 77 interface; LAPACK_EXPORTS are exported
#  (see below)
LAPACK_LIB = $(BUILD_DIR)/libsharedlapack.a

$(LAPACK_LIB) : $(BUILD_DIR)/libsharedblas.a FORCE
	$(MAKE) -C ../../lapack/F77 -f Makefile.config CONFIG=shared \
	        CONFIG_PATH=$(CURDIR)/ \
	        OUTDIR=$(abspath $(BUILD_DIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/obj-lapack \
	        ULMBLAS_INDEX_TYPE=$(SHARED_INDEX) \
	        CXXFLAGS="$(CXXFLAGS) -fPIC -DULMBLAS_REPLACEABLE_XERBLA"

$(BUILD_DIR)/functions.o : ../lbt/functions.c $(BUILD_DIR)/libsharedblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -I$(BUILD_DIR) $(F77_DEFS) \
	      -c -o $@ $<

$(BUILD_DIR)/lsame.o : lsame.c $(BUILD_DIR)/libsharedblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC \
	      -DULMBLAS_LOGICAL=$(SHARED_LOGICAL) -c -o $@ $<

#  the name reported by ulmblas_backend()
BACKEND_NAME ?= $(if $(filter ulmblas_%,$(SHARED_NAME)),$(SHARED_NAME:ulmblas_%=%),$(firstword $(subst -, ,$(TARGET_MACHINE))))

$(BUILD_DIR)/backend.o : backend.c $(BUILD_DIR)/libsharedblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC \
	      -DULMBLAS_BACKEND_NAME='"$(BACKEND_NAME)"' -c -o $@ $<

#  the hooks of FlexiBLAS (flexiblas.c)
$(BUILD_DIR)/flexiblas.o : flexiblas.c $(BUILD_DIR)/libsharedblas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC \
	      -DULMBLAS_FLEXIBLAS_INFO=$(SHARED_INDEX) -c -o $@ $<

#  The ILP64 interfaces (ILP64=1): the same makefiles and sources with the
#  configurations of config/, in a directory of their own, without the
#  assembly kernels (they are in the LP64 library); the error handlers get
#  setters of their own (ulmblas_set_f77_xerbla_64,
#  ulmblas_set_cblas_xerbla_64, see ../F77/xerbla.cc)
$(ILP64_DIR)/libshared64blas.a : FORCE
	$(MAKE) -C ../F77 -f Makefile.config CONFIG=shared64 \
	        CONFIG_PATH=$(CURDIR)/ F77_SOURCES= ULMBLAS_ASM_OBJECTS= \
	        OUTDIR=$(abspath $(ILP64_DIR)) BUILD_DIR=$(abspath $(ILP64_DIR))/obj-f77 \
	        ULMBLAS_INDEX_TYPE=$(ILP64_INDEX) \
	        CXXFLAGS="$(CXXFLAGS) -fPIC -DULMBLAS_REPLACEABLE_XERBLA \
	                  -DULMBLAS_SET_F77_XERBLA=ulmblas_set_f77_xerbla_64"

$(ILP64_DIR)/libcblas64.a : FORCE
	$(MAKE) -C ../C -f Makefile.config CONFIG=cblas64 \
	        CONFIG_PATH=$(CURDIR)/ ULMBLAS_ASM_OBJECTS= \
	        OUTDIR=$(abspath $(ILP64_DIR)) BUILD_DIR=$(abspath $(ILP64_DIR))/obj-c \
	        ULMBLAS_INDEX_TYPE=$(ILP64_INDEX) \
	        CXXFLAGS="$(CXXFLAGS) -fPIC -DULMBLAS_REPLACEABLE_XERBLA \
	                  -DULMBLAS_SET_CBLAS_XERBLA=ulmblas_set_cblas_xerbla_64"

$(ILP64_DIR)/libshared64lapack.a : $(ILP64_DIR)/libshared64blas.a FORCE
	$(MAKE) -C ../../lapack/F77 -f Makefile.config CONFIG=shared64 \
	        CONFIG_PATH=$(CURDIR)/ \
	        OUTDIR=$(abspath $(ILP64_DIR)) BUILD_DIR=$(abspath $(ILP64_DIR))/obj-lapack \
	        ULMBLAS_INDEX_TYPE=$(ILP64_INDEX) \
	        CXXFLAGS="$(CXXFLAGS) -fPIC -DULMBLAS_REPLACEABLE_XERBLA"

$(ILP64_DIR)/functions.o : ../lbt/functions.c $(ILP64_DIR)/libshared64blas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -I$(ILP64_DIR) $(F77_DEFS_64) \
	      -c -o $@ $<

$(ILP64_DIR)/lsame.o : lsame.c $(ILP64_DIR)/libshared64blas.a
	$(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC -DULMBLAS_LSAME=lsame_64_ \
	      -DULMBLAS_LOGICAL=int64_t \
	      -c -o $@ $<

ILP64_LIBS    = $(if $(filter 1,$(ILP64)),$(ILP64_DIR)/libshared64blas.a \
                                          $(ILP64_DIR)/libshared64lapack.a \
                                          $(ILP64_DIR)/libcblas64.a)
ILP64_OBJECTS = $(if $(filter 1,$(ILP64)),$(ILP64_DIR)/functions.o \
                                          $(ILP64_DIR)/lsame.o)

SHARED_OBJECTS = $(BUILD_DIR)/functions.o $(BUILD_DIR)/lsame.o \
                 $(BUILD_DIR)/backend.o $(BUILD_DIR)/flexiblas.o \
                 $(ILP64_OBJECTS)

#  Only the interfaces are exported: the Fortran 77 names (lowercase with
#  one trailing underscore, or _64_, without the helpers *_sub_ and
#  *_sub_64_), cblas_*, ulmblas_* (not the functions behind the ILP64
#  C interface, ulmblas_*_ilp64), the hooks of FlexiBLAS (flexiblas.c) and
#  the two globals of the reference CBLAS.  Not exported: the
#  C++ templates of ulmBLAS (they could clash with another library that
#  contains them) and the assembly kernels (ulmblas_openblas_*, hidden).  The list is taken from the objects (nm), as a list of
#  names (macOS) or a version script (Linux).
EXPORT_PATTERN = '^$(SYMBOL_PREFIX)([a-z0-9]+_|[a-z0-9]+_64_|xerbla_array_|xerbla_array_64_|[sg]et_blas_xerbla_|[sg]et_blas_xerbla_64_|cblas_[a-z0-9_]+|ulmblas_[a-z0-9_]+|flexiblas_backend_[gs]et_num_threads|__flexiblas_set_xerbla|RowMajorStrg|CBLAS_CallFromC)$$'
#
#  Of our LAPACK functions LAPACK_EXPORTS are exported (LP64 and ILP64),
#  by default all: a program, libulmlapack (../../lapack/reference/build.sh
#  leaves out these routines) and FlexiBLAS with the backend ulmBLAS use
#  them instead of those of Reference-LAPACK, whether faster or not.  Those
#  not in LAPACK_EXPORTS stay internal.
#
LAPACK_ROUTINES = getrf getrs gesv getri potrf potrs posv potri trtrs trtri \
                  lauum laswp
LAPACK_ALL      = $(foreach r,$(LAPACK_ROUTINES),s$(r) d$(r) c$(r) z$(r))
LAPACK_EXPORTS ?= $(LAPACK_ALL)
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
LAPACK_HIDDEN   = $(subst $(SPACE),|,$(strip \
                  $(filter-out $(LAPACK_EXPORTS),$(LAPACK_ALL))))

NOT_EXPORTED   = '_sub_$$|_sub_64_$$|^$(SYMBOL_PREFIX)ulmblas_openblas_|_ilp64$$|^$(SYMBOL_PREFIX)($(LAPACK_HIDDEN))(_|_64_)$$'

$(EXPORTS) : $(BUILD_DIR)/libsharedblas.a $(BUILD_DIR)/libcblas.a \
             $(LAPACK_LIB) $(ILP64_LIBS) $(SHARED_OBJECTS)
	$(NM) -g $^ | awk 'NF==3 && $$2 ~ /^[TDBRSVW]$$/ {print $$3}' \
	    | grep -E $(EXPORT_PATTERN) | grep -v -E $(NOT_EXPORTED) \
	    | sort -u > $@.tmp

	@case "$@" in \
	*.def) { echo EXPORTS; sed -e 's/^/    /' \
	             -e 's/^    RowMajorStrg$$/& DATA/' \
	             -e 's/^    CBLAS_CallFromC$$/& DATA/' \
	             $@.tmp; } > $@; rm -f $@.tmp ;; \
	*) if [ "$(SYMBOL_PREFIX)" = "" ]; then \
	       { echo '{ global:'; sed 's/$$/;/' $@.tmp; echo 'local: *; };'; } > $@; \
	       rm -f $@.tmp; \
	   else \
	       mv $@.tmp $@; \
	   fi ;; \
	esac

#  No C++ exception may pass through the interface (it is called from C
#  and Fortran): the objects of ulmBLAS must not refer to functions that
#  throw (operator new, std::__throw_*, __cxa_throw, ...).  Checked before
#  linking.
THROWING = '_Zn[wa][jmy](St11align_val_t)?$$|__cxa_throw$$|__cxa_rethrow$$|__cxa_allocate_exception$$|_ZSt[0-9]+__throw_'

$(SHARED_LIB) : $(BUILD_DIR)/libsharedblas.a $(BUILD_DIR)/libcblas.a \
                $(LAPACK_LIB) $(ILP64_LIBS) $(SHARED_OBJECTS) $(EXPORTS)
	@if $(NM) -u $(filter-out $(EXPORTS),$^) | grep -E $(THROWING); then \
	    echo "error: objects refer to functions that throw (above)"; \
	    exit 1; \
	fi
	@mkdir -p $(OUTDIR)
	$(CXX) $(ARCHFLAGS) $(LDFLAGS) -o $@ $(SHARED_OBJECTS) $(SHARED_LINK) \
	       $(OMPLIBS) -lm
	$(if $(SHARED_ALIAS),ln -sf $(notdir $@) $(SHARED_ALIAS))

clean :
	$(RM) -r $(BUILD_DIR) $(SHARED_LIB) $(SHARED_ALIAS) \
	      $(OUTDIR)/lib$(SHARED_NAME).dll.a
