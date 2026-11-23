#
#  The dispatching shared library (interfaces/blas/dispatch/dispatch.c):
#  libulmblas selects at load time (Windows: at the first call) a backend
#  for the CPU (central processing unit) and forwards all calls to it.
#
#      macOS, Apple Silicon   libulmblas.0.dylib (and the link
#                             libulmblas.dylib); backends libulmblas_neon,
#                             libulmblas_sme
#      Linux, x86-64          libulmblas.so.0 (and libulmblas.so); backends
#                             libulmblas_avx512, libulmblas_avx2,
#                             libulmblas_sse3, libulmblas_generic
#      Windows, x86-64        libulmblas.dll, import libraries
#      (MinGW-w64)            libulmblas.dll.a (MinGW) and ulmblas.lib (MSVC,
#                             Microsoft Visual C++);
#                             the same backends (.dll)
#
#  On macOS and Linux also libulmblas64 (libulmblas64.0.dylib,
#  libulmblas64.so.0): a second front of the same backends with their
#  interfaces with 64-bit integers under the usual names (see map64.sh),
#  and its cblas.h (D.dir/interface64/include).
#
#  The backends are shared libraries of mk/shared.mk built with other
#  ARCHFLAGS; all libraries belong in the same directory.  libulmblas
#  exports the functions of the backends (generated trampolines) and
#  ulmblas_backend() (the name of the selected backend).  Not forwarded:
#  the two variables of the reference CBLAS (C interface to the BLAS (Basic
#  Linear Algebra Subprograms)), RowMajorStrg and CBLAS_CallFromC, and the
#  error handlers xerbla_ and cblas_xerbla, and xerbla_64_ and
#  cblas_xerbla_64 of the ILP64 (64-bit integer) interfaces (see
#  dispatch.c).  The backends have the ILP64 interfaces unless ILP64=0; the
#  front then forwards them as well.
#
#  The platform is that of the compiler (TARGET_MACHINE), so cross
#  compilers work (CC, CXX, AR, RANLIB, NM, as for mk/shared.mk).  On other
#  platforms (e.g. Linux on AArch64) there is one backend only: the plain
#  shared library (make shared SHARED_VERSION=0).
#
#  $(eval $(call dispatch_lib,D)) makes the rules, set before the call:
#
#      D.dir   build directory (the backends in D.dir/BACKEND)
#      D.out   directory of the libraries
#
#  BACKEND_DIR (optional): the directory of the installed backends, for
#  packages that install them in a directory of their own (compiled into
#  the front, see ULMBLAS_BACKEND_DIR in dispatch.c).
#

DLLTOOL ?= dlltool

X86_SSE3 = -msse3 -mfpmath=sse

#  ARCHFLAGS of the backends
ARCHFLAGS_neon   =
#  sme: overridable, e.g. ARCHFLAGS_sme='-march=armv9.2-a+sme2+sme-f64f64+nosve'
#  where -mcpu is removed by the build tool (Homebrew's compiler shim)
ARCHFLAGS_sme   ?= -mcpu=apple-m4
ARCHFLAGS_sse3   = $(X86_SSE3)
#  generic: the portable kernels, for every x86-64 CPU (they all have SSE2
#  (Streaming SIMD (single instruction, multiple data) Extensions 2))
ARCHFLAGS_generic =
#  and their micro kernels for float, complex float and complex double (the
#  fastest in the campaigns 027, 030, 031, 033, 034, PERFORMANCE.md; double:
#  avx2 the own AVX (Advanced Vector Extensions) kernel, avx512 OpenBLAS
#  SKYLAKEX)
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

#  The platform: object format, backends and file name suffix; empty: no
#  dispatching library for this platform
dispatch.platform = $(if $(findstring mingw,$(TARGET_MACHINE)),mingw,$(if \
                        $(filter arm64-apple-darwin% aarch64-apple-darwin%,\
                                 $(TARGET_MACHINE)),darwin,$(if \
                        $(filter x86_64-%,$(TARGET_MACHINE)),$(if \
                        $(findstring apple-darwin,$(TARGET_MACHINE)),,elf))))
dispatch.format.mingw   = coff-x86_64
dispatch.format.darwin  = macho-arm64
dispatch.format.elf     = elf-x86_64
dispatch.backends.mingw  = sse3 avx2 avx512 generic
dispatch.backends.darwin = neon sme
dispatch.backends.elf    = sse3 avx2 avx512 generic
dispatch.so.mingw   = .dll
dispatch.so.darwin  = .dylib
dispatch.so.elf     = .so
DISPATCH_FORMAT   = $(dispatch.format.$(dispatch.platform))
DISPATCH_BACKENDS = $(dispatch.backends.$(dispatch.platform))

#  The functions of a backend, from the export list of its build (any of
#  the three formats), without the error handlers, the functions that set
#  them, the two variables and ulmblas_backend (the front has its own);
#  all backends must export the same
DISPATCH_EXCLUDED = '^(ulmblas_backend|xerbla_|cblas_xerbla|ulmblas_set_f77_xerbla|ulmblas_set_cblas_xerbla|xerbla_64_|cblas_xerbla_64|ulmblas_set_f77_xerbla_64|ulmblas_set_cblas_xerbla_64|RowMajorStrg|CBLAS_CallFromC)$$'

dispatch_functions_of = sed -e 's/[{};]/ /g' $(1) | tr ' \t' '\n\n' \
               | grep -E '^_*[A-Za-z][A-Za-z0-9_]*$$' \
               | grep -v -E '^(EXPORTS|DATA)$$' \
               | sed 's/^$(shared.symprefix)//' | grep -v -E $(DISPATCH_EXCLUDED) | sort

#  one pattern rule for both files: gen.sh runs once.  Two explicit targets
#  would be two rules with the same recipe, and make -j ran gen.sh twice at
#  the same time (a truncated trampolines.S gave a DLL (dynamic-link
#  library) without exports); grouped targets (&:) need GNU (GNU's Not
#  Unix) make 4.3, macOS has 3.81
%/names.h %/trampolines.S : %/functions.txt interfaces/blas/dispatch/gen.sh
	$(SHELL) interfaces/blas/dispatch/gen.sh $(DISPATCH_FORMAT) $< $*

%/names64.h %/trampolines64.S : %/functions64.txt interfaces/blas/dispatch/gen.sh
	$(SHELL) interfaces/blas/dispatch/gen.sh $(DISPATCH_FORMAT) $< $* 64

dispatch.ccompile = $(CC) $(CFLAGS) -O2 -fPIC
dispatch.backend_dir = $(if $(BACKEND_DIR),-DULMBLAS_BACKEND_DIR='"$(BACKEND_DIR)"')
#  the backends of libblas.so.3 and libcblas.so.3 (make sonames): where
#  they are installed (mk/install.mk: ULMBLAS_BACKENDDIR)
SONAMES_BACKEND_DIR = $(BACKEND_DIR)

define dispatch_lib
$1.backends := $$(DISPATCH_BACKENDS)
$1.so       := $$(dispatch.so.$$(dispatch.platform))

#  one backend: the shared library with the backend's ARCHFLAGS
$$(foreach b,$$($1.backends),\
    $$(eval $1.be.$$b.dir := $$($1.dir)/$$b)\
    $$(eval $1.be.$$b.out := $$($1.out))\
    $$(eval $1.be.$$b.name := ulmblas_$$b)\
    $$(eval $1.be.$$b.version :=)\
    $$(eval $1.be.$$b.i64 := 0)\
    $$(eval $1.be.$$b.ilp64 = $$$$(ILP64))\
    $$(eval $1.be.$$b.arch = $$$$(ARCHFLAGS_$$b))\
    $$(eval $$(call shared_lib,$1.be.$$b)))
$1.backend_libs := $$(foreach b,$$($1.backends),$$($1.be.$$b.lib))

ifeq ($$(dispatch.platform),mingw)
$1.lib      := $$($1.out)/libulmblas.dll
$1.alias    :=
$1.link      = -shared -Wl,--out-implib,$$($1.out)/libulmblas.dll.a \
               -static -static-libgcc
#  import library for MSVC (link.exe), from the export list
$1.implib   := $$($1.out)/ulmblas.lib
#  no libulmblas64 on Windows (see dispatch.c)
$1.lib64    :=
else ifeq ($$(dispatch.platform),darwin)
$1.lib      := $$($1.out)/libulmblas.0.dylib
$1.alias    := $$($1.out)/libulmblas.dylib
$1.link      = -dynamiclib -install_name @rpath/libulmblas.0.dylib \
               -Wl,-exported_symbols_list,$$($1.dir)/exports.txt
$1.lib64    := $$($1.out)/libulmblas64.0.dylib
$1.alias64  := $$($1.out)/libulmblas64.dylib
$1.link64    = -dynamiclib -install_name @rpath/libulmblas64.0.dylib \
               -Wl,-exported_symbols_list,$$($1.dir)/exports64.txt
else
$1.lib      := $$($1.out)/libulmblas.so.0
$1.alias    := $$($1.out)/libulmblas.so
$1.link      = -shared -Wl,-soname,libulmblas.so.0 -static-libgcc \
               -Wl,--version-script,$$($1.dir)/exports.txt -ldl
$1.lib64    := $$($1.out)/libulmblas64.so.0
$1.alias64  := $$($1.out)/libulmblas64.so
$1.link64    = -shared -Wl,-soname,libulmblas64.so.0 -static-libgcc \
               -Wl,--version-script,$$($1.dir)/exports64.txt -ldl
endif

#  libulmblas64 (macOS, Linux): a second front of the same backends, their
#  interfaces with 64-bit integers under the usual names (see map64.sh and
#  mk/shared.mk); only if the backends have them (ILP64=1); its cblas.h:
#  that of the C interface with the 64-bit index type
$1.header64 := $$(if $$($1.lib64),$$($1.dir)/interface64/include/cblas.h)
$1.all      := $$($1.lib) $$($1.backend_libs) $$($1.implib) \
               $$(if $$(filter 1,$$(ILP64)),$$($1.lib64) $$($1.header64))
lib.clean   += $$($1.dir) $$($1.lib) $$($1.alias) $$($1.lib64) \
               $$($1.alias64) $$($1.implib) $$($1.out)/libulmblas.dll.a \
               $$($1.out)/libblas.so.3 $$($1.out)/libcblas.so.3

$$($1.dir)/functions.txt : $$($1.backend_libs)
	@mkdir -p $$(@D)
	@first=; for b in $$($1.backends); do \
	    e=`ls $$($1.dir)/$$$$b/exports.txt $$($1.dir)/$$$$b/exports.def 2>/dev/null | head -1`; \
	    $$(call dispatch_functions_of,$$$$e) > $$@.$$$$b; \
	    if [ -z "$$$$first" ]; then first=$$$$b; \
	    elif [ "`cat $$@.$$$$first`" != "`cat $$@.$$$$b`" ]; then \
	        echo "error: the backends $$$$first and $$$$b export different functions"; \
	        cat $$@.$$$$first $$@.$$$$b | sort | uniq -u | head -20; \
	        exit 1; \
	    fi; \
	done; mv $$@.$$$$first $$@; rm -f $$@.*

#  the compile command of the C and assembly objects (a changed one, e.g.
#  another BACKEND_DIR, rebuilds them)
$1.ccmd := $$($1.dir)/compile-c.cmd
$$($1.ccmd) : FORCE
	@mkdir -p $$(@D)
	@$$(call record_command,$$(dispatch.ccompile) $$(dispatch.backend_dir) ; $$(CC),$$@)

#  the exports of the dispatching library (macOS: names, Linux: version
#  script; Windows: -export directives in trampolines.S, dllexport)
$$($1.dir)/exports.txt : $$($1.dir)/functions.txt
	@if [ "$$(DISPATCH_FORMAT)" = macho-arm64 ]; then \
	    { sed 's/^/_/' $$<; echo _ulmblas_backend; } > $$@; \
	else \
	    { echo '{ global:'; sed 's/$$$$/;/' $$<; echo 'ulmblas_backend;'; \
	      echo 'local: *; };'; } > $$@; \
	fi

$$($1.dir)/dispatch.o : interfaces/blas/dispatch/dispatch.c $$($1.dir)/names.h \
                        $$($1.ccmd)
	$$(dispatch.ccompile) -I$$($1.dir) $$(dispatch.backend_dir) -c -o $$@ $$<

$$($1.dir)/trampolines.o : $$($1.dir)/trampolines.S $$($1.ccmd)
	$$(CC) -c -o $$@ $$<

$$($1.lib) : $$($1.dir)/dispatch.o $$($1.dir)/trampolines.o \
             $$($1.dir)/exports.txt
	@mkdir -p $$(@D)
	$$(CC) $$(LDFLAGS) -o $$@ $$($1.dir)/dispatch.o \
	      $$($1.dir)/trampolines.o $$($1.link)
	$$(if $$($1.alias),ln -sf $$(notdir $$@) $$($1.alias))

#  libulmblas64: the exported names and those of the backends (map64.sh)
$$($1.dir)/functions64.txt : $$($1.dir)/functions.txt \
                             interfaces/blas/dispatch/map64.sh
	$$(SHELL) interfaces/blas/dispatch/map64.sh $$< > $$@

$$($1.dir)/exports64.txt : $$($1.dir)/functions64.txt
	@if [ "$$(DISPATCH_FORMAT)" = macho-arm64 ]; then \
	    { awk '{print "_" $$$$1}' $$<; echo _ulmblas_backend; } > $$@; \
	else \
	    { echo '{ global:'; awk '{print $$$$1 ";"}' $$<; \
	      echo 'ulmblas_backend;'; echo 'local: *; };'; } > $$@; \
	fi

$$($1.dir)/dispatch64.o : interfaces/blas/dispatch/dispatch.c \
                          $$($1.dir)/names64.h $$($1.ccmd)
	$$(dispatch.ccompile) -I$$($1.dir) -DULMBLAS_DISPATCH_INTERFACE64 \
	      -DULMBLAS_DISPATCH_NAMES='"names64.h"' $$(dispatch.backend_dir) \
	      -c -o $$@ $$<

$$($1.dir)/trampolines64.o : $$($1.dir)/trampolines64.S $$($1.ccmd)
	$$(CC) -c -o $$@ $$<

ifneq ($$($1.lib64),)
$$($1.lib64) : $$($1.dir)/dispatch64.o $$($1.dir)/trampolines64.o \
               $$($1.dir)/exports64.txt
	@mkdir -p $$(@D)
	$$(CC) $$(LDFLAGS) -o $$@ $$($1.dir)/dispatch64.o \
	      $$($1.dir)/trampolines64.o $$($1.link64)
	ln -sf $$(notdir $$@) $$($1.alias64)

#  its cblas.h: the header of a C interface with the 64-bit index type
#  (only the header of this build is used)
$1.h64.out   := $$($1.dir)/interface64/include
$1.h64.obj   := $$($1.dir)/interface64/obj
$1.h64.index := int64_t
$1.h64.tools := tool-ar
$$(eval $$(call cblas_lib,$1.h64,cblas))
endif

#  For distribution packages (Linux): the dispatching library once more as
#  libblas.so.3 and libcblas.so.3, the SONAMEs (shared object names) of the
#  BLAS of the distributions (Debian's and openSUSE's alternatives,
#  Gentoo's providers), so that programs built against them record these
#  names.  They are installed in a directory of their own (e.g.
#  .../x86_64-linux-gnu/ulmblas/), so they find the backends through
#  BACKEND_DIR, the directory where the backends are installed (make
#  sonames BACKEND_DIR=...; see ULMBLAS_BACKEND_DIR in dispatch.c).
$1.sonames := $$($1.out)/libblas.so.3 $$($1.out)/libcblas.so.3

$$($1.dir)/dispatch-sonames.o : interfaces/blas/dispatch/dispatch.c \
                                $$($1.dir)/names.h FORCE
	@test -n "$$(SONAMES_BACKEND_DIR)" || \
	    { echo "make sonames: BACKEND_DIR (the installed backends) is missing"; exit 2; }
	$$(dispatch.ccompile) -I$$($1.dir) \
	      -DULMBLAS_BACKEND_DIR='"$$(SONAMES_BACKEND_DIR)"' -c -o $$@ $$<

$$($1.sonames) : $$($1.dir)/dispatch-sonames.o $$($1.dir)/trampolines.o \
                 $$($1.dir)/exports.txt
	$$(CC) $$(LDFLAGS) -o $$@ $$($1.dir)/dispatch-sonames.o \
	      $$($1.dir)/trampolines.o \
	      $$(subst -soname$$(COMMA)libulmblas.so.0,-soname$$(COMMA)$$(notdir $$@),$$($1.link))

#  Windows: the exports of libulmblas.dll as a .def file, and from it the
#  import library ulmblas.lib (for MSVC; MinGW uses libulmblas.dll.a); the
#  Fortran 77 names also as DGEMM and dgemm (see gen.sh)
$$($1.dir)/exports.def : $$($1.dir)/functions.txt
	{ echo LIBRARY libulmblas.dll; echo EXPORTS; \
	  awk '{ print "    " $$$$1 } \
	       /^[a-z][a-z0-9]*_$$$$/ { f = substr($$$$1, 1, length($$$$1) - 1); \
	                              print "    " f; print "    " toupper(f) }' $$<; \
	  echo '    ulmblas_backend'; } > $$@

ifneq ($$($1.implib),)
$$($1.implib) : $$($1.dir)/exports.def $$($1.lib)
	$$(DLLTOOL) -d $$< -D libulmblas.dll -l $$@
endif
endef
