#! /bin/sh
#
#  install-shared.sh PREFIX: builds the shared libraries of ulmBLAS and
#  installs them for other programs (e.g. NumPy, interfaces/python/numpy.sh):
#
#      PREFIX/lib/libulmblas.0.dylib (.so.0, Windows: bin/libulmblas.dll
#                                          and lib/libulmblas.dll.a)
#                                          BLAS: Fortran 77 interface (LP64)
#                                          and CBLAS; on Apple Silicon and
#                                          x86-64 the front that selects a
#                                          backend (also installed:
#                                          libulmblas_sme/_neon, _avx512/
#                                          _avx2/_sse3) for the CPU
#      PREFIX/lib/libulmlapack.dylib (.so) Reference-LAPACK 3.12.1 on top of
#                                          libulmblas (needs CMake and a
#                                          Fortran compiler, FC; Windows:
#                                          bin/libulmlapack.dll,
#                                          lib/libulmlapack.dll.a,
#                                          lib/ulmlapack.lib); the sources
#                                          are downloaded unless
#                                          LAPACK_TARBALL names the tarball
#                                          (distributions: FC=none)
#      PREFIX/include/cblas.h
#      PREFIX/lib/pkgconfig/ulmblas.pc, ulmlapack.pc
#
#  (lib, bin and include: see the environment below)
#
#  Without a Fortran compiler (FC not found) only libulmblas is installed.
#  Run from anywhere; the build happens in the source tree (MAKE, default
#  gmake if found, else make).
#
#  Environment (for packages; all optional):
#
#      DESTDIR             staging directory: the files go to
#                          DESTDIR/PREFIX/..., the paths in them
#                          (pkg-config) are PREFIX/...; PREFIX must then be
#                          absolute
#      ULMBLAS_LIBDIR      the libraries (default PREFIX/lib; e.g. Debian:
#                          PREFIX/lib/x86_64-linux-gnu)
#      ULMBLAS_BINDIR      the DLLs on Windows (default PREFIX/bin)
#      ULMBLAS_INCLUDEDIR  cblas.h (default PREFIX/include; conda:
#                          PREFIX/include/ulmblas, beside the cblas.h of
#                          other packages)
#      ULMBLAS_RPATH       no: no -Wl,-rpath in the pkg-config files (e.g.
#                          Fedora; never on Windows)
#      ULMBLAS_VERSION     the version for pkg-config (default: the last
#                          release tag, the file VERSION.txt of a release
#                          tarball, or the directory name ulmBLAS-X.Y.Z)
#      ULMBLAS_STATIC_LIBSTDCXX
#                          no (Linux): libstdc++ linked dynamically (by
#                          default statically if libstdc++.a exists)
#      ULMBLAS_BACKENDDIR  the backends of the dispatching library (default
#                          LIBDIR; distributions: a directory of their
#                          own, e.g. LIBDIR/ulmblas, whose path is then
#                          compiled into the dispatching library)
#      ULMBLAS_SONAMES     yes (Linux, x86-64): also libblas.so.3 and
#                          libcblas.so.3, the dispatching library with the
#                          SONAMEs of the distributions' BLAS (for their
#                          alternatives), in ULMBLAS_SONAMEDIR (default
#                          LIBDIR/ulmblas); they load the backends from
#                          BACKENDDIR
#      ULMBLAS_STEP        build: only build libulmblas (and the libraries
#                          of ULMBLAS_SONAMES), install nothing (the build
#                          step of a package; its install step runs the
#                          script again with the same environment)
#
set -u
[ $# -eq 1 ] || { echo "usage: install-shared.sh PREFIX" >&2; exit 2; }
DESTDIR=${DESTDIR:-}
if [ -n "$DESTDIR" ]; then
    case $1 in
        /*) PREFIX=$1 ;;
        *)  echo "error: with DESTDIR the PREFIX must be absolute" >&2
            exit 2 ;;
    esac
elif [ "${ULMBLAS_STEP:-}" = build ]; then
    #  the build only: PREFIX need not exist yet
    case $1 in
        /*) PREFIX=$1 ;;
        *)  PREFIX=$(pwd)/$1 ;;
    esac
else
    mkdir -p "$1" || exit 1
    PREFIX=$(cd "$1" && pwd)
fi
TOP=$(cd "`dirname "$0"`/.." && pwd)
FC=${FC:-gfortran}
#  MAKE may contain options (make -j8); it is not passed on in the
#  environment, where it would replace $(MAKE) of the sub-makes (each with
#  its own -j instead of the job server of the first make)
MAKECMD=${MAKE:-}
unset MAKE
if [ -z "$MAKECMD" ]; then
    if command -v gmake > /dev/null 2>&1; then MAKECMD=gmake; else MAKECMD=make; fi
fi
LIBDIR=${ULMBLAS_LIBDIR:-$PREFIX/lib}
BINDIR=${ULMBLAS_BINDIR:-$PREFIX/bin}
INCDIR=${ULMBLAS_INCLUDEDIR:-$PREFIX/include}

#  the version for pkg-config (v0.3.0 -> 0.3.0): the last release tag, else
#  the file VERSION.txt (filled in by git archive, see .gitattributes),
#  else the name of the directory of a release tarball (ulmBLAS-0.3.0)
VERSION=${ULMBLAS_VERSION:-}
if [ -z "$VERSION" ]; then
    VERSION=$(git -C "$TOP" describe --tags --abbrev=0 2> /dev/null)
fi
if [ -z "$VERSION" ] && [ -f "$TOP/VERSION.txt" ]; then
    VERSION=$(sed -n '1{/[$]Format/d;p;}' "$TOP/VERSION.txt")
fi
if [ -z "$VERSION" ]; then
    VERSION=$(basename "$TOP" | sed -n 's/^ulmBLAS-//p')
fi
VERSION=$(echo "${VERSION:-0}" | sed 's/^v//')

#  the libraries (BIN: DLLs on Windows, beside the programs; LIB: all
#  others, links included; BACKENDS: the backends of the dispatching
#  library, installed in BACKENDDIR)
BIN=
BACKENDS=
#  ULMBLAS_PLATFORM (e.g. Linux-x86_64): the target platform when cross
#  compiling (conda-forge: from $target_platform); default: this machine
PLATFORM=${ULMBLAS_PLATFORM:-`uname -s`-`uname -m`}
case $PLATFORM in
    Darwin-arm64)
        SO=dylib; TARGET=shared-dispatch
        LIB="libulmblas.0.dylib libulmblas.dylib"
        BACKENDS="libulmblas_neon.dylib libulmblas_sme.dylib" ;;
    Linux-x86_64)
        SO=so; TARGET=shared-dispatch
        LIB="libulmblas.so.0 libulmblas.so"
        BACKENDS="libulmblas_sse3.so libulmblas_avx2.so libulmblas_avx512.so
                  libulmblas_generic.so" ;;
    MINGW*-x86_64|MSYS*-x86_64)
        SO=dll; TARGET=shared-dispatch
        BIN="libulmblas.dll libulmblas_sse3.dll libulmblas_avx2.dll
             libulmblas_avx512.dll libulmblas_generic.dll"
        LIB="libulmblas.dll.a ulmblas.lib" ;;
    Darwin-*)
        echo "error: on macOS only Apple Silicon is supported" >&2; exit 2 ;;
    *)  #  one backend: the micro kernels of ARCHFLAGS (default of make.inc)
        SO=so; TARGET="shared SHARED_VERSION=0"
        LIB="libulmblas.so.0 libulmblas.so" ;;
esac

cd "$TOP" || exit 1
#  the makefiles take BUILD_DIR and OUTDIR from the environment, where
#  build tools may have set them for themselves (rattler-build: BUILD_DIR)
unset BUILD_DIR OUTDIR
MAKEVARS=
if [ "${ULMBLAS_STATIC_LIBSTDCXX:-yes}" = no ]; then
    MAKEVARS="STATIC_LIBSTDCXX="
fi
#  (ignored where there are no backends, e.g. Linux on aarch64)
BACKENDDIR=${ULMBLAS_BACKENDDIR:-$LIBDIR}
if [ -n "$BACKENDS" ] && [ "$BACKENDDIR" != "$LIBDIR" ]; then
    #  compiled into the dispatching library (see dispatch.c)
    MAKEVARS="$MAKEVARS BACKEND_DIR=$BACKENDDIR"
fi
$MAKECMD $TARGET $MAKEVARS || exit 1
SONAMES=${ULMBLAS_SONAMES:-no}
if [ "$SONAMES" = yes ]; then
    case $PLATFORM in
        Linux-x86_64) ;;
        *) echo "error: ULMBLAS_SONAMES=yes only on Linux (x86-64)" >&2
           exit 2 ;;
    esac
    $MAKECMD -C interfaces/blas/dispatch sonames $MAKEVARS \
        BACKEND_DIR="$BACKENDDIR" || exit 1
fi
if [ "${ULMBLAS_STEP:-}" = build ]; then
    echo "built libulmblas (ULMBLAS_STEP=build: nothing installed)"
    exit 0
fi
mkdir -p "$DESTDIR$LIBDIR/pkgconfig" "$DESTDIR$INCDIR" || exit 1
for l in $LIB; do
    cp -fP "$l" "$DESTDIR$LIBDIR/" || exit 1
done
if [ -n "$BACKENDS" ]; then
    mkdir -p "$DESTDIR$BACKENDDIR" || exit 1
    cp -f $BACKENDS "$DESTDIR$BACKENDDIR/" || exit 1
fi
if [ -n "$BIN" ]; then
    mkdir -p "$DESTDIR$BINDIR" || exit 1
    for l in $BIN; do
        cp -f "$l" "$DESTDIR$BINDIR/" || exit 1
    done
fi
if [ "$SONAMES" = yes ]; then
    SONAMEDIR=${ULMBLAS_SONAMEDIR:-$LIBDIR/ulmblas}
    mkdir -p "$DESTDIR$SONAMEDIR" || exit 1
    cp -f libblas.so.3 libcblas.so.3 "$DESTDIR$SONAMEDIR/" || exit 1
fi
#  -Wl,-rpath in the pkg-config files (not on Windows, not with
#  ULMBLAS_RPATH=no)
RPATH=' -Wl,-rpath,${libdir}'
if [ "${ULMBLAS_RPATH:-yes}" = no ] || [ "$SO" = dll ]; then
    RPATH=
fi
#  cblas.h: generated in the build of the (first) library
H=`ls interfaces/blas/dispatch/build.dispatch/*/cblas.h \
       interfaces/blas/shared/build.shared/cblas.h 2> /dev/null | head -1`
cp -f "$H" "$DESTDIR$INCDIR/" || exit 1

cat > "$DESTDIR$LIBDIR/pkgconfig/ulmblas.pc" << EOF
libdir=$LIBDIR
includedir=$INCDIR

Name: ulmblas
Description: ulmBLAS (Fortran 77 LP64 and CBLAS interface)
Version: $VERSION
Libs: -L\${libdir} -lulmblas$RPATH
Cflags: -I\${includedir}
EOF
echo "installed libulmblas in $DESTDIR$LIBDIR"

if [ "$FC" = none ] || ! command -v "$FC" > /dev/null 2>&1; then
    echo "no Fortran compiler ($FC): libulmlapack not built" >&2
    exit 0
fi
FC=$FC sh interfaces/lapack/reference/build.sh . build/lapack || exit 1
if [ "$SO" = dll ]; then
    cp -f libulmlapack.dll "$DESTDIR$BINDIR/" || exit 1
    cp -f libulmlapack.dll.a ulmlapack.lib "$DESTDIR$LIBDIR/" || exit 1
else
    cp -f "libulmlapack.$SO" "$DESTDIR$LIBDIR/" || exit 1
fi
cat > "$DESTDIR$LIBDIR/pkgconfig/ulmlapack.pc" << EOF
libdir=$LIBDIR

Name: ulmlapack
Description: Reference-LAPACK 3.12.1 on top of ulmBLAS
Version: 3.12.1
Requires: ulmblas
Libs: -L\${libdir} -lulmlapack$RPATH
Cflags:
EOF
echo "installed libulmlapack in $DESTDIR$LIBDIR"
