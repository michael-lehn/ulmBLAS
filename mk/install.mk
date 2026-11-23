#
#  Build and install the shared libraries for other programs (e.g. NumPy,
#  interfaces/python/numpy.sh), packages and distributions:
#
#      make build-shared PREFIX=dir     build (PREFIX: compiled into the
#                                       dispatching library, see below)
#      make check-package               the Netlib test programs against
#                                       them (packaging/check.sh; FC)
#      make install-shared PREFIX=dir   build and install
#
#  Installed:
#
#      PREFIX/lib/libulmblas.0.dylib (.so.0, Windows: bin/libulmblas.dll
#                                          and lib/libulmblas.dll.a) BLAS
#                                          (Basic Linear Algebra Subprograms):
#                                          Fortran 77 interface with LP64
#                                          (32-bit integer) indices and CBLAS
#                                          (C interface to the BLAS); on Apple
#                                          Silicon and x86-64 the front that
#                                          selects a backend (also installed:
#                                          libulmblas_sme/_neon,
#                                          _avx512/_avx2/_sse3/_generic) for
#                                          the CPU (central processing unit)
#      PREFIX/lib/libulmlapack.dylib (.so) Reference-LAPACK 3.12.1 (the
#                                          reference implementation of LAPACK
#                                          (Linear Algebra PACKage)) on top of
#                                          libulmblas (needs a Fortran
#                                          compiler, FC; Windows:
#                                          bin/libulmlapack.dll,
#                                          lib/libulmlapack.dll.a,
#                                          lib/ulmlapack.lib); the sources are
#                                          downloaded unless LAPACK_TARBALL
#                                          names the tarball (distributions:
#                                          FC=none)
#      PREFIX/lib/libulmblas64.0.dylib (.so.0; not on Windows)
#                                          BLAS with 64-bit integers under
#                                          the usual names (dgemm_,
#                                          cblas_dgemm; as Debian's
#                                          libblas64, e.g. for FlexiBLAS
#                                          with 64-bit integers); on Apple
#                                          Silicon and x86-64 a front of
#                                          the same backends
#      PREFIX/include/cblas.h
#      PREFIX/include/ulmblas64/cblas.h    the one of libulmblas64
#      PREFIX/lib/pkgconfig/ulmblas.pc, ulmblas64.pc, ulmlapack.pc
#
#  The platform is that of the compiler (TARGET_MACHINE; cross compilers
#  work).  Without a Fortran compiler (FC=none or not found) libulmlapack
#  is not built.
#
#  Settings (for packages; all optional; make variables or environment,
#  the names of interfaces/install-shared.sh):
#
#      DESTDIR             staging directory: the files go to
#                          DESTDIR/PREFIX/..., the paths in them
#                          (pkg-config) are PREFIX/...; PREFIX must then be
#                          absolute
#      ULMBLAS_LIBDIR      the libraries (default PREFIX/lib; e.g. Debian:
#                          PREFIX/lib/x86_64-linux-gnu)
#      ULMBLAS_BINDIR      the DLLs (dynamic-link libraries) on Windows
#                          (default PREFIX/bin)
#      ULMBLAS_INCLUDEDIR  cblas.h (default PREFIX/include; conda:
#                          PREFIX/include/ulmblas, beside the cblas.h of
#                          other packages)
#      ULMBLAS_RPATH       no: no -Wl,-rpath in the pkg-config files (e.g.
#                          Fedora; never on Windows)
#      ULMBLAS_VERSION     the version for pkg-config (default: the last
#                          release tag, the file VERSION.txt of a release
#                          tarball, or the directory name ulmBLAS-X.Y.Z)
#      ULMBLAS_STATIC_LIBSTDCXX
#                          no (Linux, GCC (the GNU Compiler Collection)
#                          before 13): no -static-libstdc++;
#                          the libraries link no C++ runtime (clang and
#                          GCC 13 or newer: -nostdlib++, no effect)
#      ULMBLAS_BACKENDDIR  the backends of the dispatching library (default
#                          LIBDIR; distributions: a directory of their
#                          own, e.g. LIBDIR/ulmblas, whose path is then
#                          compiled into the dispatching library)
#      ULMBLAS_SONAMES     yes (Linux): also libblas.so.3 and libcblas.so.3,
#                          the library with the SONAMEs (shared object names)
#                          of the distributions' BLAS (for their alternatives),
#                          in ULMBLAS_SONAMEDIR (default LIBDIR/ulmblas); on
#                          x86-64 the dispatching library, which loads the
#                          backends from BACKENDDIR
#      ULMBLAS_INTERFACE64 no: without libulmblas64
#

install.libdir := $(or $(ULMBLAS_LIBDIR),$(PREFIX)/lib)
install.bindir := $(or $(ULMBLAS_BINDIR),$(PREFIX)/bin)
install.incdir := $(or $(ULMBLAS_INCLUDEDIR),$(PREFIX)/include)
install.backenddir := $(or $(ULMBLAS_BACKENDDIR),$(install.libdir))
install.sonamedir  := $(or $(ULMBLAS_SONAMEDIR),$(install.libdir)/ulmblas)
install.goals := $(filter build-shared install-shared,$(MAKECMDGOALS))

ifeq ($(ULMBLAS_STATIC_LIBSTDCXX),no)
STATIC_LIBSTDCXX :=
endif

#  the backends: compiled into the dispatching library if they are not
#  installed beside it (see dispatch.c); libblas.so.3 and libcblas.so.3
#  are installed in a directory of their own and always get the directory
#  of the backends
ifneq ($(ULMBLAS_BACKENDDIR),)
ifneq ($(ULMBLAS_BACKENDDIR),$(install.libdir))
BACKEND_DIR ?= $(ULMBLAS_BACKENDDIR)
endif
endif
ifneq ($(install.goals),)
SONAMES_BACKEND_DIR = $(or $(BACKEND_DIR),$(install.backenddir))
endif

#
#  What is built and installed, per platform: install.lib (the libraries,
#  links included), install.bin (DLLs, Windows), install.backends (in
#  BACKENDDIR), install.lib64 and install.header64 (libulmblas64), the
#  header cblas.h, the library libulmlapack is linked with
#
install.i64 := $(if $(filter no,$(ULMBLAS_INTERFACE64)),,yes)
install.sonames := $(if $(filter yes,$(ULMBLAS_SONAMES)),yes)

ifneq ($(dispatch.platform),)
install.backend.libs := $(top.dispatch.backend_libs)
install.header := $(top.dispatch.be.$(firstword $(sort $(top.dispatch.backends))).c.header)
ifeq ($(dispatch.platform),mingw)
install.so       := dll
install.bin      := $(top.dispatch.lib) $(install.backend.libs)
install.lib      := ./libulmblas.dll.a ./ulmblas.lib
install.backends :=
install.build    := $(install.bin) $(top.dispatch.implib)
else
install.so       := $(if $(filter darwin,$(dispatch.platform)),dylib,so)
install.lib      := $(top.dispatch.lib) $(top.dispatch.alias)
install.backends := $(install.backend.libs)
install.build    := $(top.dispatch.lib) $(install.backends)
ifeq ($(install.i64)$(filter 1,$(ILP64)),yes1)
install.lib64    := $(top.dispatch.lib64) $(top.dispatch.alias64)
install.header64 := $(top.dispatch.header64)
install.build    += $(top.dispatch.lib64) $(install.header64)
endif
endif
else
#  one library: the micro kernels of ARCHFLAGS (default of make.inc), with
#  the version in the name
install.so     := so
install.header := $(top.shared.c.header)
install.lib    := $(top.shared.lib) $(top.shared.alias)
install.build  := $(top.shared.lib)
ifeq ($(install.i64),yes)
install.lib64    := $(top.shared64.lib) $(top.shared64.alias)
install.header64 := $(top.shared64.c.header)
install.build    += $(top.shared64.lib)
endif
endif

#  libblas.so.3 and libcblas.so.3 (ULMBLAS_SONAMES=yes)
ifeq ($(install.sonames),yes)
ifneq ($(filter mingw darwin,$(shared.platform)),)
install.error := ULMBLAS_SONAMES=yes only on Linux
endif
install.sonamelibs := ./libblas.so.3 ./libcblas.so.3
install.build      += $(install.sonamelibs)
endif

#  libulmlapack: Reference-LAPACK on top of the library, with LAPACK's own
#  makefiles (interfaces/lapack/reference/build.sh), if a Fortran compiler
#  is found; not for check-package
install.blas := $(install.build)
ifneq ($(install.goals),)
install.fc := $(if $(filter none,$(FC)),,$(shell command -v $(FC) >/dev/null 2>&1 && echo yes))
endif
ifeq ($(install.fc),yes)
install.lapack := ./libulmlapack.$(install.so)
install.build  += $(install.lapack)
#  the build of LAPACK runs its own make (third-party makefiles); +: it
#  gets the job server
$(install.lapack) : $(install.blas) interfaces/lapack/reference/build.sh
	+FC='$(FC)' $(SHELL) interfaces/lapack/reference/build.sh . build/lapack
endif

install.rpath = $(if $(or $(filter no,$(ULMBLAS_RPATH)),$(filter dll,$(install.so))),, -Wl$(COMMA)-rpath$(COMMA)$${libdir})

#  the version for pkg-config (v0.3.0 -> 0.3.0): the last release tag, else
#  the file VERSION.txt (filled in by git archive, see .gitattributes),
#  else the name of the directory of a release tarball (ulmBLAS-0.3.0)
install.version = $(patsubst v%,%,$(or $(ULMBLAS_VERSION),$(shell \
    git -C $(TOPDIR) describe --tags --abbrev=0 2>/dev/null),$(shell \
    sed -n '1{/[$$]Format/d;p;}' $(TOPDIR)/VERSION.txt 2>/dev/null),$(patsubst \
    ulmBLAS-%,%,$(filter ulmBLAS-%,$(notdir $(CURDIR)))),0))

ifneq ($(install.goals)$(filter check-package,$(MAKECMDGOALS)),)
ifneq ($(install.error),)
$(error $(install.error))
endif
endif
ifneq ($(filter install-shared,$(MAKECMDGOALS)),)
ifeq ($(PREFIX),)
$(error usage: make install-shared PREFIX=dir)
endif
ifneq ($(DESTDIR),)
ifeq ($(filter /%,$(PREFIX)),)
$(error with DESTDIR the PREFIX must be absolute)
endif
endif
endif

build-shared : $(install.build)

#  the BLAS libraries only (no libulmlapack)
check-package : $(install.blas)
	FC='$(if $(filter none,$(FC)),gfortran,$(FC))' CC='$(CC)' \
	    $(SHELL) packaging/check.sh

install.dest = $(DESTDIR)$(install.libdir)
install-shared : build-shared
	mkdir -p "$(install.dest)/pkgconfig" "$(DESTDIR)$(install.incdir)"
	cp -fP $(install.lib) "$(install.dest)/"
	$(if $(install.backends),mkdir -p "$(DESTDIR)$(install.backenddir)")
	$(if $(install.backends),cp -f $(install.backends) "$(DESTDIR)$(install.backenddir)/")
	$(if $(install.bin),mkdir -p "$(DESTDIR)$(install.bindir)")
	$(if $(install.bin),cp -f $(install.bin) "$(DESTDIR)$(install.bindir)/")
	$(if $(install.sonamelibs),mkdir -p "$(DESTDIR)$(install.sonamedir)")
	$(if $(install.sonamelibs),cp -f $(install.sonamelibs) "$(DESTDIR)$(install.sonamedir)/")
	cp -f $(install.header) "$(DESTDIR)$(install.incdir)/"
	{ echo 'libdir=$(install.libdir)'; \
	  echo 'includedir=$(install.incdir)'; \
	  echo; \
	  echo 'Name: ulmblas'; \
	  echo 'Description: ulmBLAS (Fortran 77 LP64 and CBLAS interface)'; \
	  echo 'Version: $(install.version)'; \
	  echo 'Libs: -L$${libdir} -lulmblas$(install.rpath)'; \
	  echo 'Cflags: -I$${includedir}'; \
	} > "$(install.dest)/pkgconfig/ulmblas.pc"
	@echo "installed libulmblas in $(install.dest)"
ifneq ($(install.lib64),)
	cp -fP $(install.lib64) "$(install.dest)/"
	mkdir -p "$(DESTDIR)$(install.incdir)/ulmblas64"
	cp -f $(install.header64) "$(DESTDIR)$(install.incdir)/ulmblas64/cblas.h"
	{ echo 'libdir=$(install.libdir)'; \
	  echo 'includedir=$(install.incdir)/ulmblas64'; \
	  echo; \
	  echo 'Name: ulmblas64'; \
	  echo 'Description: ulmBLAS (Fortran 77 and CBLAS interface with 64-bit integers, usual names)'; \
	  echo 'Version: $(install.version)'; \
	  echo 'Libs: -L$${libdir} -lulmblas64$(install.rpath)'; \
	  echo 'Cflags: -I$${includedir}'; \
	} > "$(install.dest)/pkgconfig/ulmblas64.pc"
	@echo "installed libulmblas64 in $(install.dest)"
endif
ifneq ($(install.lapack),)
ifeq ($(install.so),dll)
	cp -f libulmlapack.dll "$(DESTDIR)$(install.bindir)/"
	cp -f libulmlapack.dll.a ulmlapack.lib "$(install.dest)/"
else
	cp -f $(install.lapack) "$(install.dest)/"
endif
	{ echo 'libdir=$(install.libdir)'; \
	  echo; \
	  echo 'Name: ulmlapack'; \
	  echo 'Description: Reference-LAPACK 3.12.1 on top of ulmBLAS'; \
	  echo 'Version: 3.12.1'; \
	  echo 'Requires: ulmblas'; \
	  echo 'Libs: -L$${libdir} -lulmlapack$(install.rpath)'; \
	  echo 'Cflags:'; \
	} > "$(install.dest)/pkgconfig/ulmlapack.pc"
	@echo "installed libulmlapack in $(install.dest)"
else
	@echo "no Fortran compiler ($(FC)): libulmlapack not built"
endif
