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
#                                          Fortran compiler, FC)
#      PREFIX/include/cblas.h
#      PREFIX/lib/pkgconfig/ulmblas.pc, ulmlapack.pc
#
#  Without a Fortran compiler (FC not found) only libulmblas is installed.
#  Run from anywhere; the build happens in the source tree (MAKE, default
#  gmake if found, else make).
#
set -u
[ $# -eq 1 ] || { echo "usage: install-shared.sh PREFIX" >&2; exit 2; }
mkdir -p "$1/lib/pkgconfig" "$1/include" || exit 1
PREFIX=$(cd "$1" && pwd)
TOP=$(cd "`dirname "$0"`/.." && pwd)
FC=${FC:-gfortran}
if [ -z "${MAKE:-}" ]; then
    if command -v gmake > /dev/null 2>&1; then MAKE=gmake; else MAKE=make; fi
fi
#  the version for pkg-config: the last release tag (v0.3.0 -> 0.3.0)
VERSION=$(git -C "$TOP" describe --tags --abbrev=0 2> /dev/null | sed 's/^v//')
VERSION=${VERSION:-0}

#  the libraries (BIN: DLLs on Windows, beside the programs; LIB: all
#  others, links included)
BIN=
#  ULMBLAS_PLATFORM (e.g. Linux-x86_64): the target platform when cross
#  compiling (conda-forge: from $target_platform); default: this machine
PLATFORM=${ULMBLAS_PLATFORM:-`uname -s`-`uname -m`}
#  ULMBLAS_INCLUDEDIR: where cblas.h goes (default PREFIX/include; conda:
#  PREFIX/include/ulmblas, beside the cblas.h of other packages)
INCDIR=${ULMBLAS_INCLUDEDIR:-$PREFIX/include}
case $PLATFORM in
    Darwin-arm64)
        SO=dylib; TARGET=shared-dispatch
        LIB="libulmblas.0.dylib libulmblas.dylib libulmblas_neon.dylib
             libulmblas_sme.dylib" ;;
    Linux-x86_64)
        SO=so; TARGET=shared-dispatch
        LIB="libulmblas.so.0 libulmblas.so libulmblas_sse3.so
             libulmblas_avx2.so libulmblas_avx512.so" ;;
    MINGW*-x86_64|MSYS*-x86_64)
        SO=dll; TARGET=shared-dispatch
        BIN="libulmblas.dll libulmblas_sse3.dll libulmblas_avx2.dll
             libulmblas_avx512.dll"
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
$MAKE $TARGET || exit 1
for l in $LIB; do
    cp -fP "$l" "$PREFIX/lib/" || exit 1
done
if [ -n "$BIN" ]; then
    mkdir -p "$PREFIX/bin" || exit 1
    for l in $BIN; do
        cp -f "$l" "$PREFIX/bin/" || exit 1
    done
fi
#  cblas.h: generated in the build of the (first) library
H=`ls interfaces/blas/dispatch/build.dispatch/*/cblas.h \
       interfaces/blas/shared/build.shared/cblas.h 2> /dev/null | head -1`
mkdir -p "$INCDIR" && cp -f "$H" "$INCDIR/" || exit 1

cat > "$PREFIX/lib/pkgconfig/ulmblas.pc" << EOF
libdir=$PREFIX/lib
includedir=$INCDIR

Name: ulmblas
Description: ulmBLAS (Fortran 77 LP64 and CBLAS interface)
Version: $VERSION
Libs: -L\${libdir} -lulmblas -Wl,-rpath,\${libdir}
Cflags: -I\${includedir}
EOF
echo "installed libulmblas in $PREFIX/lib"

if [ "$SO" = dll ]; then
    echo "LAPACK on Windows: not yet built by this script" >&2
    exit 0
fi
if [ "$FC" = none ] || ! command -v "$FC" > /dev/null 2>&1; then
    echo "no Fortran compiler ($FC): libulmlapack not built" >&2
    exit 0
fi
FC=$FC sh interfaces/lapack/reference/build.sh . build/lapack || exit 1
cp -f "libulmlapack.$SO" "$PREFIX/lib/" || exit 1
cat > "$PREFIX/lib/pkgconfig/ulmlapack.pc" << EOF
libdir=$PREFIX/lib

Name: ulmlapack
Description: Reference-LAPACK 3.12.1 on top of ulmBLAS
Version: 3.12.1
Requires: ulmblas
Libs: -L\${libdir} -lulmlapack -Wl,-rpath,\${libdir}
Cflags:
EOF
echo "installed libulmlapack in $PREFIX/lib"
