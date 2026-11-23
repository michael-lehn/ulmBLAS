#
#  Shared library with the standard BLAS (Basic Linear Algebra
#  Subprograms) interfaces, for programs and packages that link a system
#  BLAS (NumPy, SciPy, R, FlexiBLAS, ...):
#
#      libulmblas.so      (Linux)
#      libulmblas.dylib   (macOS)
#      libulmblas.dll     (Windows, MinGW-w64; import library libulmblas.dll.a)
#
#  $(eval $(call shared_lib,S)) makes the rules of a shared library S, set
#  before the call:
#
#      S.dir      build directory (objects, interfaces, export list)
#      S.out      directory of the library
#      S.name     the library is lib$(S.name).so / .dylib / .dll
#      S.version  with the version in the name: libulmblas.so.0 (soname),
#                 libulmblas.0.dylib (install name), and a link without it;
#                 empty: no version (the backends of mk/dispatch.mk)
#      S.i64      1: the interfaces with 64-bit integers under the usual
#                 names (libulmblas64, see below)
#      S.ilp64    1: also the interfaces with 64-bit integers and the
#                 suffix _64_ (see below); not with S.i64=1
#      S.arch     ARCHFLAGS (select the micro kernels)
#
#  The platform is that of the compiler (TARGET_MACHINE), so cross
#  compilers work (e.g. CXX=x86_64-w64-mingw32-g++
#  CC=x86_64-w64-mingw32-gcc NM=x86_64-w64-mingw32-nm
#  AR=x86_64-w64-mingw32-ar RANLIB=x86_64-w64-mingw32-ranlib).
#
#  The library contains the Fortran 77 interface with LP64 (32-bit
#  integer) indices and the usual names (dgemm_, ddot_, xerbla_, lsame_,
#  ...) and the C interface (CBLAS (C interface to the BLAS): cblas_dgemm,
#  ..., cblas_xerbla, and the ulmblas_ functions including GEMM (general
#  matrix-matrix product) with arbitrary strides), and the LAPACK (Linear
#  Algebra PACKage) functions of ulmBLAS (LAPACK_EXPORTS: ?getrf, ?potrf,
#  ...; the libulmlapack of interfaces/lapack/reference uses them instead
#  of its own routines).  The objects are built with -fPIC; the Fortran
#  functions (DDOT, ...) come from interfaces/blas/lbt/functions.c, so no
#  Fortran compiler is needed.  The return values of the complex functions
#  (ZDOTC, ...) follow gfortran.
#
#  ILP64=1 (default): the library also contains both interfaces with
#  ILP64 (64-bit integer) indices and the names of the reference BLAS and
#  CBLAS built with BUILD_INDEX64_EXT_API (Arch, openSUSE): dgemm_64_, ...,
#  xerbla_64_, xerbla_array_64_, lsame_64_ and cblas_dgemm_64, ...,
#  cblas_xerbla_64.  The LAPACK of these distributions (liblapack.so.3)
#  calls them.  The same sources, built a second time with
#  ULMBLAS_INDEX_TYPE=int64_t; the functions behind the interfaces are the
#  same (instantiated with std::ptrdiff_t).  ILP64=0: LP64 only.
#
#  The headers of the two interfaces are generated in S.dir: cblas.h (and
#  ulmblas_strided.h) and sharedblas.h (F77 prototypes); those of the ILP64
#  interfaces in S.dir/ilp64 (cblas64.h, shared64blas.h).
#
#  S.i64=1: libulmblas64 instead, the interfaces with 64-bit integers
#  under the usual names (dgemm_, cblas_dgemm, lsame_ with a 64-bit
#  LOGICAL, xerbla_, ...), as Debian's libblas64: the library for programs
#  compiled with 64-bit INTEGER (gfortran -fdefault-integer-8) and the
#  backend of FlexiBLAS with 64-bit integers (libflexiblas64), which looks
#  up only these names.  Without the LP64 interfaces and without the names
#  with _64_; its headers (cblas.h, sharedblas.h) have the 64-bit index
#  type.  A program must not load libulmblas and libulmblas64 together:
#  the names are the same, the integers are not.  On macOS and Linux on
#  x86-64, mk/dispatch.mk builds libulmblas64 as a front of the backends of
#  libulmblas.
#

ILP64 ?= 1

#  The C++ runtime (Linux): no object refers to it (NO_CXX_RUNTIME in
#  make.inc), the libraries are linked with -nostdlib++; GCC (the GNU
#  Compiler Collection) before 13 does not know it, then
#  -static-libstdc++ (adds nothing), or if libstdc++.a is missing (RHEL
#  (Red Hat Enterprise Linux): package libstdc++-static) or
#  STATIC_LIBSTDCXX= (ULMBLAS_STATIC_LIBSTDCXX=no, mk/install.mk)
#  -Wl,--as-needed, which drops the unused libstdc++.so.
STATIC_LIBSTDCXX = $(eval STATIC_LIBSTDCXX := $(if $(filter /%,$(shell \
                       $(CXX) -print-file-name=libstdc++.a)),-static-libstdc++))$(STATIC_LIBSTDCXX)
NO_LIBSTDCXX     = $(or $(NOSTDLIBXX),$(STATIC_LIBSTDCXX),-Wl$(COMMA)--as-needed)

#  The platform of the shared libraries: mingw, darwin or elf
shared.platform = $(if $(findstring mingw,$(TARGET_MACHINE)),mingw,$(if \
                      $(findstring apple-darwin,$(TARGET_MACHINE)),darwin,elf))

#
#  Only the interfaces are exported: the Fortran 77 names (lowercase with
#  one trailing underscore, or _64_, without the helpers *_sub_ and
#  *_sub_64_), cblas_*, ulmblas_* (not the functions behind the ILP64 C
#  interface, ulmblas_*_ilp64), the hooks of FlexiBLAS (flexiblas.c) and
#  the two globals of the reference CBLAS.  Not exported: the C++
#  templates of ulmBLAS (they could clash with another library that
#  contains them) and the assembly kernels (ulmblas_openblas_*, hidden).
#  The list is taken from the objects (nm), as a list of names (macOS), a
#  version script (Linux) or a .def file (Windows).
#
#  Of our LAPACK functions LAPACK_EXPORTS are exported (LP64 and ILP64),
#  by default all: a program, libulmlapack (interfaces/lapack/reference/
#  build.sh leaves out these routines) and FlexiBLAS with the backend
#  ulmBLAS use them instead of those of Reference-LAPACK, whether faster or
#  not.  Those not in LAPACK_EXPORTS stay internal.
#
LAPACK_ROUTINES = getrf getrs gesv getri potrf potrs posv potri trtrs trtri \
                  lauum laswp
LAPACK_ALL      = $(foreach r,$(LAPACK_ROUTINES),s$(r) d$(r) c$(r) z$(r))
LAPACK_EXPORTS ?= $(LAPACK_ALL)
LAPACK_HIDDEN   = $(subst $(SPACE),|,$(strip \
                  $(filter-out $(LAPACK_EXPORTS),$(LAPACK_ALL))))

shared.symprefix = $(if $(filter darwin,$(shared.platform)),_)
EXPORT_PATTERN   = '^$(shared.symprefix)([a-z0-9]+_|[a-z0-9]+_64_|xerbla_array_|xerbla_array_64_|[sg]et_blas_xerbla_|[sg]et_blas_xerbla_64_|cblas_[a-z0-9_]+|ulmblas_[a-z0-9_]+|flexiblas_backend_[gs]et_num_threads|__flexiblas_set_xerbla|RowMajorStrg|CBLAS_CallFromC)$$'
NOT_EXPORTED     = '_sub_$$|_sub_64_$$|^$(shared.symprefix)ulmblas_openblas_|_ilp64$$|^$(shared.symprefix)($(LAPACK_HIDDEN))(_|_64_)$$'

#  the flags of the objects of a shared library (the ILP64 interfaces add
#  the setters of their error handlers, ulmblas_set_f77_xerbla_64 and
#  ulmblas_set_cblas_xerbla_64, see interfaces/blas/F77/xerbla.cc)
shared.cxxflags = -fPIC -DULMBLAS_REPLACEABLE_XERBLA

#  the C objects: interfaces/blas/lbt/functions.c with the F77 header of
#  the build, lsame.c, backend.c (the name ulmblas_backend() reports),
#  flexiblas.c (the hooks of FlexiBLAS)
shared.ccompile = $(CC) $(ULMBLAS_CFLAGS) $(CFLAGS) -fPIC

define shared_lib
$1.index   := $$(if $$(filter 1,$$($1.i64)),int64_t,int)
$1.ilp64x  := $$(if $$(filter 1,$$($1.i64)),0,$$($1.ilp64))
$1.idir    := $$($1.dir)/ilp64

#  the interfaces with 32-bit integers (or 64-bit ones, S.i64=1): F77
#  (with the assembly kernels), C and LAPACK
$1.f77.out      := $$($1.dir)
$1.f77.obj      := $$($1.dir)/obj-f77
$1.f77.f        :=
$1.f77.index    := $$($1.index)
$1.f77.arch      = $$($1.arch)
$1.f77.cxxflags := $$(shared.cxxflags)
$1.f77.asm      := yes
$1.f77.tools    := tool-ar
$$(eval $$(call f77blas_lib,$1.f77,shared))

$1.c.out        := $$($1.dir)
$1.c.obj        := $$($1.dir)/obj-c
$1.c.index      := $$($1.index)
$1.c.arch        = $$($1.arch)
$1.c.cxxflags   := $$(shared.cxxflags)
$1.c.tools      := tool-ar
$$(eval $$(call cblas_lib,$1.c,cblas))

$1.lapack.out      := $$($1.dir)
$1.lapack.obj      := $$($1.dir)/obj-lapack
$1.lapack.index    := $$($1.index)
$1.lapack.arch      = $$($1.arch)
$1.lapack.cxxflags := $$(shared.cxxflags)
$1.lapack.blas     := $1.f77
$1.lapack.tools    := tool-ar
$$(eval $$(call f77lapack_lib,$1.lapack,shared))

$1.libs    := $$($1.f77.lib) $$($1.c.lib) $$($1.lapack.lib)
$1.objects := $$($1.dir)/functions.o $$($1.dir)/lsame.o \
              $$($1.dir)/backend.o $$($1.dir)/flexiblas.o

#  the compile command of the C objects (a changed one rebuilds them)
$1.ccmd := $$($1.dir)/compile-c.cmd
$$($1.ccmd) : FORCE | tool-ar
	@mkdir -p $$(@D)
	@$$(call record_command,$$(shared.ccompile),$$@)

#  the name reported by ulmblas_backend()
$1.backend := $$(if $$(filter ulmblas_%,$$($1.name)),$$($1.name:ulmblas_%=%),$$(firstword $$(subst -, ,$$(TARGET_MACHINE))))

$$($1.dir)/functions.o : interfaces/blas/lbt/functions.c $$($1.f77.header) \
                         $$($1.ccmd)
	$$(shared.ccompile) -I$$($1.dir) -D'BLAS_HEADER=<sharedblas.h>' \
	      -D'F77BLAS(x)=x$$(PASTE)_' -c -o $$@ $$<

$$($1.dir)/lsame.o : interfaces/blas/shared/lsame.c $$($1.ccmd) \
                     | $$($1.f77.header)
	$$(shared.ccompile) -DULMBLAS_LOGICAL=$$($1.index) -c -o $$@ $$<

$$($1.dir)/backend.o : interfaces/blas/shared/backend.c $$($1.ccmd) \
                       | $$($1.f77.header)
	$$(shared.ccompile) -DULMBLAS_BACKEND_NAME='"$$($1.backend)"' -c -o $$@ $$<

$$($1.dir)/flexiblas.o : interfaces/blas/shared/flexiblas.c $$($1.ccmd) \
                         | $$($1.f77.header)
	$$(shared.ccompile) -DULMBLAS_FLEXIBLAS_INFO=$$($1.index) -c -o $$@ $$<

#  the interfaces with 64-bit integers and the suffix _64_ (S.ilp64=1), in
#  a directory of their own, without the assembly kernels (they are in the
#  LP64 library)
ifeq ($$($1.ilp64x),1)
$1.f77_64.out      := $$($1.idir)
$1.f77_64.obj      := $$($1.idir)/obj-f77
$1.f77_64.f        :=
$1.f77_64.index    := int64_t
$1.f77_64.arch      = $$($1.arch)
$1.f77_64.cxxflags := $$(shared.cxxflags) -DULMBLAS_SET_F77_XERBLA=ulmblas_set_f77_xerbla_64
$1.f77_64.tools    := tool-ar
$$(eval $$(call f77blas_lib,$1.f77_64,shared64))

$1.c64.out         := $$($1.idir)
$1.c64.obj         := $$($1.idir)/obj-c
$1.c64.index       := int64_t
$1.c64.arch         = $$($1.arch)
$1.c64.cxxflags    := $$(shared.cxxflags) -DULMBLAS_SET_CBLAS_XERBLA=ulmblas_set_cblas_xerbla_64
$1.c64.tools       := tool-ar
$$(eval $$(call cblas_lib,$1.c64,cblas64))

$1.lapack64.out      := $$($1.idir)
$1.lapack64.obj      := $$($1.idir)/obj-lapack
$1.lapack64.index    := int64_t
$1.lapack64.arch      = $$($1.arch)
$1.lapack64.cxxflags := $$(shared.cxxflags)
$1.lapack64.blas     := $1.f77_64
$1.lapack64.tools    := tool-ar
$$(eval $$(call f77lapack_lib,$1.lapack64,shared64))

$1.libs    += $$($1.f77_64.lib) $$($1.lapack64.lib) $$($1.c64.lib)
$1.objects += $$($1.idir)/functions.o $$($1.idir)/lsame.o

$$($1.idir)/functions.o : interfaces/blas/lbt/functions.c \
                          $$($1.f77_64.header) $$($1.ccmd)
	$$(shared.ccompile) -I$$($1.idir) -D'BLAS_HEADER=<shared64blas.h>' \
	      -D'F77BLAS(x)=x$$(PASTE)_64_' -c -o $$@ $$<

$$($1.idir)/lsame.o : interfaces/blas/shared/lsame.c $$($1.ccmd) \
                      | $$($1.f77_64.header)
	$$(shared.ccompile) -DULMBLAS_LSAME=lsame_64_ -DULMBLAS_LOGICAL=int64_t \
	      -c -o $$@ $$<
endif

#  the library and its link command, per platform
ifeq ($$(shared.platform),mingw)
$1.exports := $$($1.dir)/exports.def
$1.lib     := $$($1.out)/lib$$($1.name).dll
$1.alias   :=
$1.link     = -shared -Wl,--out-implib,$$($1.out)/lib$$($1.name).dll.a \
              $$($1.exports) -Wl,--whole-archive $$($1.libs) \
              -Wl,--no-whole-archive \
              -static $$(or $$(NOSTDLIBXX),-static-libstdc++) -static-libgcc \
              -Wl,--no-undefined
else ifeq ($$(shared.platform),darwin)
$1.exports := $$($1.dir)/exports.txt
$1.file    := lib$$($1.name)$$(if $$($1.version),.$$($1.version)).dylib
$1.lib     := $$($1.out)/$$($1.file)
$1.alias   := $$(if $$($1.version),$$($1.out)/lib$$($1.name).dylib)
$1.link     = -dynamiclib -install_name @rpath/$$($1.file) \
              $$(foreach l,$$($1.libs),-Wl$$(COMMA)-force_load$$(COMMA)$$(l)) \
              -Wl,-exported_symbols_list,$$($1.exports) $$(NOSTDLIBXX)
else
$1.exports := $$($1.dir)/exports.txt
$1.file    := lib$$($1.name).so$$(if $$($1.version),.$$($1.version))
$1.lib     := $$($1.out)/$$($1.file)
$1.alias   := $$(if $$($1.version),$$($1.out)/lib$$($1.name).so)
$1.link     = -shared -Wl,-soname,$$($1.file) \
              -Wl,--whole-archive $$($1.libs) -Wl,--no-whole-archive \
              $$(NO_LIBSTDCXX) -static-libgcc -pthread -Wl,--no-undefined \
              -Wl,--version-script,$$($1.exports)
endif
lib.clean += $$($1.dir) $$($1.lib) $$($1.alias) $$($1.out)/lib$$($1.name).dll.a

$$($1.objects) : | tool-ar

$$($1.exports) : $$($1.libs) $$($1.objects)
	$$(NM) -g $$^ | awk 'NF==3 && $$$$2 ~ /^[TDBRSVW]$$$$/ {print $$$$3}' \
	    | grep -E $$(EXPORT_PATTERN) | grep -v -E $$(NOT_EXPORTED) \
	    | sort -u > $$@.tmp
	@case "$$@" in \
	*.def) { echo EXPORTS; sed -e 's/^/    /' \
	             -e 's/^    RowMajorStrg$$$$/& DATA/' \
	             -e 's/^    CBLAS_CallFromC$$$$/& DATA/' \
	             $$@.tmp; } > $$@; rm -f $$@.tmp ;; \
	*) if [ "$$(shared.symprefix)" = "" ]; then \
	       { echo '{ global:'; sed 's/$$$$/;/' $$@.tmp; echo 'local: *; };'; } > $$@; \
	       rm -f $$@.tmp; \
	   else \
	       mv $$@.tmp $$@; \
	   fi ;; \
	esac

#  The objects must not refer to the C++ runtime (NO_CXX_RUNTIME in
#  make.inc; this also excludes functions that throw: no C++ exception may
#  pass through the interface, it is called from C and Fortran).  Checked
#  before linking.  The headers of the C interface belong to the build
#  (interfaces/install-shared, packaging/check.sh).
$$($1.lib) : $$($1.libs) $$($1.objects) $$($1.exports) | $$($1.c.all)
	@$$(NO_CXX_RUNTIME) $$($1.libs) $$($1.objects)
	@mkdir -p $$(@D)
	$$(CXX) $$($1.arch) $$(LDFLAGS) -o $$@ $$($1.objects) $$($1.link) \
	       $$(OMPLIBS) -lm
	$$(if $$($1.alias),ln -sf $$(notdir $$@) $$($1.alias))

#  S.sonames (Linux): the same objects linked once more under these names
#  and SONAMEs (shared object names), libblas.so.3 and libcblas.so.3: the
#  alternatives of the distributions on platforms without the dispatching
#  library
ifneq ($$(strip $$($1.sonames)),)
$$(addprefix $$($1.out)/,$$($1.sonames)) : $$($1.libs) $$($1.objects) \
                                           $$($1.exports) $$($1.lib)
	$$(CXX) $$($1.arch) $$(LDFLAGS) -o $$@ $$($1.objects) \
	       $$(subst -soname$$(COMMA)$$($1.file),-soname$$(COMMA)$$(notdir $$@),$$($1.link)) \
	       $$(OMPLIBS) -lm
lib.clean += $$(addprefix $$($1.out)/,$$($1.sonames))
endif
endef
