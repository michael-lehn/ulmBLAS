#
#  The dispatching shared library (see dispatch.c): libulmblas selects at
#  load time (Windows: at the first call) a backend for the CPU and
#  forwards all calls to it.
#
#      macOS, Apple Silicon   libulmblas.0.dylib (and the link
#                             libulmblas.dylib); backends libulmblas_neon,
#                             libulmblas_sme
#      Linux, x86-64          libulmblas.so.0 (and libulmblas.so); backends
#                             libulmblas_avx512, libulmblas_avx2,
#                             libulmblas_sse3, libulmblas_generic
#      Windows, x86-64        libulmblas.dll, import libraries
#      (MinGW-w64)            libulmblas.dll.a (MinGW) and ulmblas.lib (MSVC);
#                             the same backends (.dll)
#
#  On macOS and Linux also libulmblas64 (libulmblas64.0.dylib,
#  libulmblas64.so.0): a second front of the same backends with their
#  interfaces with 64-bit integers under the usual names (see map64.sh),
#  and its cblas.h (BUILD_DIR/interface64/include).
#
#  The backends are the shared library of ../shared built with other
#  ARCHFLAGS; all libraries belong in the same directory.  libulmblas
#  exports the functions of the backends (generated trampolines) and
#  ulmblas_backend() (the name of the selected backend).  Not forwarded:
#  the two variables of the reference CBLAS (RowMajorStrg, CBLAS_CallFromC)
#  and the error handlers xerbla_ and cblas_xerbla, and xerbla_64_ and
#  cblas_xerbla_64 of the ILP64 interfaces (see dispatch.c).  The
#  backends have the ILP64 interfaces unless they are built with ILP64=0
#  (../shared); the front then forwards them as well.
#
#  The platform is that of the compiler ($(CXX) -dumpmachine), so cross
#  compilers work (CC, CXX, AR, RANLIB, NM, as for ../shared).  On other
#  platforms (e.g. Linux on AArch64) there is one backend only: build the
#  plain shared library (make shared SHARED_VERSION=0).
#
TOPDIR = ../../..

include $(TOPDIR)/make.inc

OUTDIR     ?= $(TOPDIR)
BUILD_DIR  ?= build.dispatch
NM         ?= nm
DLLTOOL    ?= dlltool

TARGET_MACHINE := $(shell $(CXX) -dumpmachine)

X86_SSE3 = -msse3 -mfpmath=sse

ifneq (,$(findstring mingw,$(TARGET_MACHINE)))
FORMAT         = coff-x86_64
BACKENDS       = sse3 avx2 avx512 generic
SO             = .dll
DISPATCH_LIB   = $(OUTDIR)/libulmblas.dll
DISPATCH_ALIAS =
DISPATCH_LINK  = -shared -Wl,--out-implib,$(OUTDIR)/libulmblas.dll.a \
                 -static -static-libgcc
#  import library for MSVC (link.exe), from the export list
DISPATCH_IMPLIB = $(OUTDIR)/ulmblas.lib
SYMBOL_PREFIX  =
#  no libulmblas64 on Windows (see dispatch.c)
DISPATCH64_LIB =
else ifneq (,$(findstring apple-darwin,$(TARGET_MACHINE)))
FORMAT         = macho-arm64
BACKENDS       = neon sme
SO             = .dylib
DISPATCH_LIB   = $(OUTDIR)/libulmblas.0.dylib
DISPATCH_ALIAS = $(OUTDIR)/libulmblas.dylib
DISPATCH_LINK  = -dynamiclib -install_name @rpath/libulmblas.0.dylib \
                 -Wl,-exported_symbols_list,$(BUILD_DIR)/exports.txt
DISPATCH64_LIB   = $(OUTDIR)/libulmblas64.0.dylib
DISPATCH64_ALIAS = $(OUTDIR)/libulmblas64.dylib
DISPATCH64_LINK  = -dynamiclib -install_name @rpath/libulmblas64.0.dylib \
                   -Wl,-exported_symbols_list,$(BUILD_DIR)/exports64.txt
SYMBOL_PREFIX  = _
else ifneq (,$(filter x86_64-%,$(TARGET_MACHINE)))
FORMAT         = elf-x86_64
BACKENDS       = sse3 avx2 avx512 generic
SO             = .so
DISPATCH_LIB   = $(OUTDIR)/libulmblas.so.0
DISPATCH_ALIAS = $(OUTDIR)/libulmblas.so
DISPATCH_LINK  = -shared -Wl,-soname,libulmblas.so.0 -static-libgcc \
                 -Wl,--version-script,$(BUILD_DIR)/exports.txt -ldl
DISPATCH64_LIB   = $(OUTDIR)/libulmblas64.so.0
DISPATCH64_ALIAS = $(OUTDIR)/libulmblas64.so
DISPATCH64_LINK  = -shared -Wl,-soname,libulmblas64.so.0 -static-libgcc \
                   -Wl,--version-script,$(BUILD_DIR)/exports64.txt -ldl
SYMBOL_PREFIX  =
else ifeq (,$(filter clean,$(MAKECMDGOALS)))
$(error no dispatching library for $(TARGET_MACHINE): use make shared SHARED_VERSION=0)
endif

#  ARCHFLAGS of the backends
ARCHFLAGS_neon   =
#  sme: overridable, e.g. ARCHFLAGS_sme='-march=armv9.2-a+sme2+sme-f64f64+nosve'
#  where -mcpu is removed by the build tool (Homebrew's compiler shim)
ARCHFLAGS_sme   ?= -mcpu=apple-m4
ARCHFLAGS_sse3   = $(X86_SSE3)
#  generic: the portable kernels, for every x86-64 CPU (SSE2)
ARCHFLAGS_generic =
#  and their micro kernels for float, complex float and complex double (the
#  fastest in the campaigns 027, 030, 031, 033, 034, PERFORMANCE.md; double:
#  avx2 the own AVX kernel, avx512 OpenBLAS SKYLAKEX)
ARCHFLAGS_avx2   = $(X86_SSE3) -mavx -mfma -mavx2 \
                   -DULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4 \
                   -DULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2 \
                   -DULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2
ARCHFLAGS_avx512 = $(X86_SSE3) -mavx -mfma -mavx2 -mavx512f -mavx512dq \
                   -mavx512vl \
                   -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12 \
                   -DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4 \
                   -DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2 \
                   -DULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4

BACKEND_LIBS = $(foreach b,$(BACKENDS),$(OUTDIR)/libulmblas_$(b)$(SO))

#  libulmblas64 (macOS, Linux): a second front of the same backends, their
#  interfaces with 64-bit integers under the usual names (see map64.sh and
#  ../shared, INTERFACE64); only if the backends have them (ILP64=1, the
#  default of ../shared)
ILP64 ?= 1
DISPATCH64 = $(if $(filter 1,$(ILP64)),$(DISPATCH64_LIB) $(HEADER64))
#  its cblas.h: that of the C interface with the 64-bit index type
HEADER64   = $(if $(DISPATCH64_LIB),$(BUILD_DIR)/interface64/include/cblas.h)

.DELETE_ON_ERROR :
.PHONY : all clean sonames FORCE

COMMA := ,

all : $(DISPATCH_LIB) $(BACKEND_LIBS) $(DISPATCH_IMPLIB) $(DISPATCH64)

#  one backend: the library of ../shared with the backend's ARCHFLAGS
$(OUTDIR)/libulmblas_%$(SO) : FORCE
	$(MAKE) -C ../shared SHARED_NAME=ulmblas_$* \
	        ARCHFLAGS="$(ARCHFLAGS_$*)" NM="$(NM)" \
	        OUTDIR=$(abspath $(OUTDIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/$*

#  The functions of a backend, from the export list of its build (any of
#  the three formats), without the error handlers, the functions that set
#  them, the two variables and ulmblas_backend (the front has its own);
#  all backends must export the same
EXCLUDED = '^(ulmblas_backend|xerbla_|cblas_xerbla|ulmblas_set_f77_xerbla|ulmblas_set_cblas_xerbla|xerbla_64_|cblas_xerbla_64|ulmblas_set_f77_xerbla_64|ulmblas_set_cblas_xerbla_64|RowMajorStrg|CBLAS_CallFromC)$$'

functions_of = sed -e 's/[{};]/ /g' $(1) | tr ' \t' '\n\n' \
               | grep -E '^_*[A-Za-z][A-Za-z0-9_]*$$' \
               | grep -v -E '^(EXPORTS|DATA)$$' \
               | sed 's/^$(SYMBOL_PREFIX)//' | grep -v -E $(EXCLUDED) | sort

$(BUILD_DIR)/functions.txt : $(BACKEND_LIBS)
	@mkdir -p $(BUILD_DIR)
	@first=; for b in $(BACKENDS); do \
	    e=`ls $(BUILD_DIR)/$$b/exports.txt $(BUILD_DIR)/$$b/exports.def 2>/dev/null | head -1`; \
	    $(call functions_of,$$e) > $@.$$b; \
	    if [ -z "$$first" ]; then first=$$b; \
	    elif [ "`cat $@.$$first`" != "`cat $@.$$b`" ]; then \
	        echo "error: the backends $$first and $$b export different functions"; \
	        cat $@.$$first $@.$$b | sort | uniq -u | head -20; \
	        exit 1; \
	    fi; \
	done; mv $@.$$first $@; rm -f $@.*

$(BUILD_DIR)/names.h $(BUILD_DIR)/trampolines.S : $(BUILD_DIR)/functions.txt \
                                                   gen.sh
	$(SHELL) gen.sh $(FORMAT) $< $(BUILD_DIR)

#  the exports of the dispatching library (macOS: names, Linux: version
#  script; Windows: -export directives in trampolines.S, dllexport)
$(BUILD_DIR)/exports.txt : $(BUILD_DIR)/functions.txt
	@if [ "$(FORMAT)" = macho-arm64 ]; then \
	    { sed 's/^/_/' $<; echo _ulmblas_backend; } > $@; \
	else \
	    { echo '{ global:'; sed 's/$$/;/' $<; echo 'ulmblas_backend;'; \
	      echo 'local: *; };'; } > $@; \
	fi

#  BACKEND_DIR (optional): the directory of the installed backends, for
#  packages that install them in a directory of their own
$(BUILD_DIR)/dispatch.o : dispatch.c $(BUILD_DIR)/names.h
	$(CC) $(CFLAGS) -O2 -fPIC -I$(BUILD_DIR) \
	      $(if $(BACKEND_DIR),-DULMBLAS_BACKEND_DIR='"$(BACKEND_DIR)"') \
	      -c -o $@ $<

$(BUILD_DIR)/trampolines.o : $(BUILD_DIR)/trampolines.S
	$(CC) -c -o $@ $<

$(DISPATCH_LIB) : $(BUILD_DIR)/dispatch.o $(BUILD_DIR)/trampolines.o \
                  $(BUILD_DIR)/exports.txt
	$(CC) $(LDFLAGS) -o $@ $(BUILD_DIR)/dispatch.o \
	      $(BUILD_DIR)/trampolines.o $(DISPATCH_LINK)
	$(if $(DISPATCH_ALIAS),ln -sf $(notdir $@) $(DISPATCH_ALIAS))

#  libulmblas64: the exported names and those of the backends (map64.sh)
$(BUILD_DIR)/functions64.txt : $(BUILD_DIR)/functions.txt map64.sh
	$(SHELL) map64.sh $< > $@

$(BUILD_DIR)/names64.h $(BUILD_DIR)/trampolines64.S : \
                $(BUILD_DIR)/functions64.txt gen.sh
	$(SHELL) gen.sh $(FORMAT) $< $(BUILD_DIR) 64

$(BUILD_DIR)/exports64.txt : $(BUILD_DIR)/functions64.txt
	@if [ "$(FORMAT)" = macho-arm64 ]; then \
	    { awk '{print "_" $$1}' $<; echo _ulmblas_backend; } > $@; \
	else \
	    { echo '{ global:'; awk '{print $$1 ";"}' $<; \
	      echo 'ulmblas_backend;'; echo 'local: *; };'; } > $@; \
	fi

$(BUILD_DIR)/dispatch64.o : dispatch.c $(BUILD_DIR)/names64.h
	$(CC) $(CFLAGS) -O2 -fPIC -I$(BUILD_DIR) -DULMBLAS_DISPATCH_INTERFACE64 \
	      -DULMBLAS_DISPATCH_NAMES='"names64.h"' \
	      $(if $(BACKEND_DIR),-DULMBLAS_BACKEND_DIR='"$(BACKEND_DIR)"') \
	      -c -o $@ $<

$(BUILD_DIR)/trampolines64.o : $(BUILD_DIR)/trampolines64.S
	$(CC) -c -o $@ $<

$(DISPATCH64_LIB) : $(BUILD_DIR)/dispatch64.o $(BUILD_DIR)/trampolines64.o \
                    $(BUILD_DIR)/exports64.txt
	$(CC) $(LDFLAGS) -o $@ $(BUILD_DIR)/dispatch64.o \
	      $(BUILD_DIR)/trampolines64.o $(DISPATCH64_LINK)
	ln -sf $(notdir $@) $(DISPATCH64_ALIAS)

$(BUILD_DIR)/interface64/include/cblas.h : FORCE
	$(MAKE) -C ../C -f Makefile.config CONFIG=cblas ULMBLAS_ASM_OBJECTS= \
	        OUTDIR=$(abspath $(BUILD_DIR))/interface64/include \
	        BUILD_DIR=$(abspath $(BUILD_DIR))/interface64/obj \
	        ULMBLAS_INDEX_TYPE=int64_t $(abspath $@)

#  For distribution packages (Linux): the dispatching library once more as
#  libblas.so.3 and libcblas.so.3, the SONAMEs of the BLAS of the
#  distributions (Debian's and openSUSE's alternatives, Gentoo's
#  providers), so that programs built against them record these names.
#  They are installed in a directory of their own (e.g.
#  .../x86_64-linux-gnu/ulmblas/), so they find the backends through
#  BACKEND_DIR, the directory where the backends are installed (make
#  sonames BACKEND_DIR=...; see ULMBLAS_BACKEND_DIR in dispatch.c).
#
SONAME_LIBS = $(OUTDIR)/libblas.so.3 $(OUTDIR)/libcblas.so.3

sonames : $(SONAME_LIBS) $(BACKEND_LIBS)

$(BUILD_DIR)/dispatch-sonames.o : dispatch.c $(BUILD_DIR)/names.h FORCE
	@test -n "$(BACKEND_DIR)" || \
	    { echo "make sonames: BACKEND_DIR (the installed backends) is missing"; exit 2; }
	$(CC) $(CFLAGS) -O2 -fPIC -I$(BUILD_DIR) \
	      -DULMBLAS_BACKEND_DIR='"$(BACKEND_DIR)"' -c -o $@ $<

$(SONAME_LIBS) : $(BUILD_DIR)/dispatch-sonames.o $(BUILD_DIR)/trampolines.o \
                 $(BUILD_DIR)/exports.txt
	$(CC) $(LDFLAGS) -o $@ $(BUILD_DIR)/dispatch-sonames.o \
	      $(BUILD_DIR)/trampolines.o \
	      $(subst -soname$(COMMA)libulmblas.so.0,-soname$(COMMA)$(notdir $@),$(DISPATCH_LINK))

#  Windows: the exports of libulmblas.dll as a .def file, and from it the
#  import library ulmblas.lib (for MSVC; MinGW uses libulmblas.dll.a); the
#  Fortran 77 names also as DGEMM and dgemm (see gen.sh)
$(BUILD_DIR)/exports.def : $(BUILD_DIR)/functions.txt
	{ echo LIBRARY libulmblas.dll; echo EXPORTS; \
	  awk '{ print "    " $$1 } \
	       /^[a-z][a-z0-9]*_$$/ { f = substr($$1, 1, length($$1) - 1); \
	                              print "    " f; print "    " toupper(f) }' $<; \
	  echo '    ulmblas_backend'; } > $@

$(OUTDIR)/ulmblas.lib : $(BUILD_DIR)/exports.def $(DISPATCH_LIB)
	$(DLLTOOL) -d $< -D libulmblas.dll -l $@

clean :
	$(RM) -r $(BUILD_DIR) $(DISPATCH_LIB) $(DISPATCH_ALIAS) $(BACKEND_LIBS) \
	      $(DISPATCH64_LIB) $(DISPATCH64_ALIAS) \
	      $(OUTDIR)/libblas.so.3 $(OUTDIR)/libcblas.so.3 \
	      $(OUTDIR)/libulmblas.dll.a $(DISPATCH_IMPLIB) \
	      $(foreach b,$(BACKENDS),$(OUTDIR)/libulmblas_$(b).dll.a)
