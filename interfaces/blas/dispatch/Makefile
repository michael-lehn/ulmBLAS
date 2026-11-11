#
#  The dispatching shared library (see dispatch.c): libulmblas selects at
#  load time (Windows: at the first call) a backend for the CPU and
#  forwards all calls to it.
#
#      macOS, Apple Silicon   libulmblas.0.dylib (and the link
#                             libulmblas.dylib); backends libulmblas_neon,
#                             libulmblas_sme
#      Linux, x86-64          libulmblas.so.0 (and libulmblas.so); backends
#                             libulmblas_sse3, libulmblas_avx2,
#                             libulmblas_avx512
#      Windows, x86-64        libulmblas.dll, import libraries
#      (MinGW-w64)            libulmblas.dll.a (MinGW) and ulmblas.lib (MSVC);
#                             the same backends (.dll)
#
#  The backends are the shared library of ../shared built with other
#  ARCHFLAGS; all libraries belong in the same directory.  libulmblas
#  exports the functions of the backends (generated trampolines) and
#  ulmblas_backend() (the name of the selected backend).  Not forwarded:
#  the two variables of the reference CBLAS (RowMajorStrg, CBLAS_CallFromC)
#  and the error handlers xerbla_ and cblas_xerbla (see dispatch.c).
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
BACKENDS       = sse3 avx2 avx512
SO             = .dll
DISPATCH_LIB   = $(OUTDIR)/libulmblas.dll
DISPATCH_ALIAS =
DISPATCH_LINK  = -shared -Wl,--out-implib,$(OUTDIR)/libulmblas.dll.a \
                 -static -static-libgcc
#  import library for MSVC (link.exe), from the export list
DISPATCH_IMPLIB = $(OUTDIR)/ulmblas.lib
SYMBOL_PREFIX  =
else ifneq (,$(findstring apple-darwin,$(TARGET_MACHINE)))
FORMAT         = macho-arm64
BACKENDS       = neon sme
SO             = .dylib
DISPATCH_LIB   = $(OUTDIR)/libulmblas.0.dylib
DISPATCH_ALIAS = $(OUTDIR)/libulmblas.dylib
DISPATCH_LINK  = -dynamiclib -install_name @rpath/libulmblas.0.dylib \
                 -Wl,-exported_symbols_list,$(BUILD_DIR)/exports.txt
SYMBOL_PREFIX  = _
else ifneq (,$(filter x86_64-%,$(TARGET_MACHINE)))
FORMAT         = elf-x86_64
BACKENDS       = sse3 avx2 avx512
SO             = .so
DISPATCH_LIB   = $(OUTDIR)/libulmblas.so.0
DISPATCH_ALIAS = $(OUTDIR)/libulmblas.so
DISPATCH_LINK  = -shared -Wl,-soname,libulmblas.so.0 -static-libgcc \
                 -Wl,--version-script,$(BUILD_DIR)/exports.txt -ldl
SYMBOL_PREFIX  =
else
$(error no dispatching library for $(TARGET_MACHINE): use make shared SHARED_VERSION=0)
endif

#  ARCHFLAGS of the backends
ARCHFLAGS_neon   =
ARCHFLAGS_sme    = -mcpu=apple-m4
ARCHFLAGS_sse3   = $(X86_SSE3)
ARCHFLAGS_avx2   = $(X86_SSE3) -mavx -mfma -mavx2
ARCHFLAGS_avx512 = $(X86_SSE3) -mavx -mfma -mavx2 -mavx512f \
                   -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12

BACKEND_LIBS = $(foreach b,$(BACKENDS),$(OUTDIR)/libulmblas_$(b)$(SO))

.DELETE_ON_ERROR :
.PHONY : all clean FORCE

all : $(DISPATCH_LIB) $(BACKEND_LIBS) $(DISPATCH_IMPLIB)

#  one backend: the library of ../shared with the backend's ARCHFLAGS
$(OUTDIR)/libulmblas_%$(SO) : FORCE
	$(MAKE) -C ../shared SHARED_NAME=ulmblas_$* \
	        ARCHFLAGS="$(ARCHFLAGS_$*)" NM="$(NM)" \
	        OUTDIR=$(abspath $(OUTDIR)) BUILD_DIR=$(abspath $(BUILD_DIR))/$*

#  The functions of a backend, from the export list of its build (any of
#  the three formats), without the error handlers, the functions that set
#  them, the two variables and ulmblas_backend (the front has its own);
#  all backends must export the same
EXCLUDED = '^(ulmblas_backend|xerbla_|cblas_xerbla|ulmblas_set_f77_xerbla|ulmblas_set_cblas_xerbla|RowMajorStrg|CBLAS_CallFromC)$$'

functions_of = sed -e 's/[{};]/ /g' $(1) | tr ' \t' '\n\n' \
               | grep -E '^_?[A-Za-z][A-Za-z0-9_]*$$' \
               | grep -v -E '^(EXPORTS|DATA)$$' \
               | sed 's/^$(SYMBOL_PREFIX)//' | grep -v -E $(EXCLUDED) | sort

$(BUILD_DIR)/functions.txt : $(BACKEND_LIBS)
	@mkdir -p $(BUILD_DIR)
	@first=; for b in $(BACKENDS); do \
	    e=`ls $(BUILD_DIR)/$$b/exports.txt $(BUILD_DIR)/$$b/exports.def 2>/dev/null | head -1`; \
	    $(call functions_of,$$e) > $@.$$b; \
	    if [ -z "$$first" ]; then first=$$b; \
	    elif ! cmp -s $@.$$first $@.$$b; then \
	        echo "error: the backends $$first and $$b export different functions"; \
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

$(BUILD_DIR)/dispatch.o : dispatch.c $(BUILD_DIR)/names.h
	$(CC) $(CFLAGS) -O2 -fPIC -I$(BUILD_DIR) -c -o $@ $<

$(BUILD_DIR)/trampolines.o : $(BUILD_DIR)/trampolines.S
	$(CC) -c -o $@ $<

$(DISPATCH_LIB) : $(BUILD_DIR)/dispatch.o $(BUILD_DIR)/trampolines.o \
                  $(BUILD_DIR)/exports.txt
	$(CC) $(LDFLAGS) -o $@ $(BUILD_DIR)/dispatch.o \
	      $(BUILD_DIR)/trampolines.o $(DISPATCH_LINK)
	$(if $(DISPATCH_ALIAS),ln -sf $(notdir $@) $(DISPATCH_ALIAS))

#  Windows: the exports of libulmblas.dll as a .def file, and from it the
#  import library ulmblas.lib (for MSVC; MinGW uses libulmblas.dll.a)
$(BUILD_DIR)/exports.def : $(BUILD_DIR)/functions.txt
	{ echo LIBRARY libulmblas.dll; echo EXPORTS; \
	  sed 's/^/    /' $<; echo '    ulmblas_backend'; } > $@

$(OUTDIR)/ulmblas.lib : $(BUILD_DIR)/exports.def $(DISPATCH_LIB)
	$(DLLTOOL) -d $< -D libulmblas.dll -l $@

clean :
	$(RM) -r $(BUILD_DIR) $(DISPATCH_LIB) $(DISPATCH_ALIAS) $(BACKEND_LIBS) \
	      $(OUTDIR)/libulmblas.dll.a $(DISPATCH_IMPLIB) \
	      $(foreach b,$(BACKENDS),$(OUTDIR)/libulmblas_$(b).dll.a)
