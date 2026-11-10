#! /bin/sh
#
#  install-shared.sh PREFIX: builds the shared libraries of ulmBLAS and
#  installs them for other programs (e.g. NumPy, interfaces/python/numpy.sh):
#
#      PREFIX/lib/libulmblas.dylib (.so)   BLAS: Fortran 77 interface (LP64)
#                                          and CBLAS.  On macOS with Apple
#                                          Silicon the front that selects
#                                          libulmblas_sme or libulmblas_neon
#                                          (also installed) at load time
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

case `uname -s`-`uname -m` in
    Darwin-arm64) SO=dylib; TARGET=shared-dispatch;
                  LIBS="libulmblas libulmblas_neon libulmblas_sme" ;;
    Darwin-*)     SO=dylib; TARGET=shared; LIBS=libulmblas ;;
    *)            SO=so;    TARGET=shared; LIBS=libulmblas ;;
esac

cd "$TOP" || exit 1
$MAKE $TARGET || exit 1
for l in $LIBS; do
    cp -f "$l.$SO" "$PREFIX/lib/" || exit 1
done
cp -f interfaces/blas/shared/build.shared/cblas.h "$PREFIX/include/" || exit 1

cat > "$PREFIX/lib/pkgconfig/ulmblas.pc" << EOF
libdir=$PREFIX/lib
includedir=$PREFIX/include

Name: ulmblas
Description: ulmBLAS (Fortran 77 LP64 and CBLAS interface)
Version: $VERSION
Libs: -L\${libdir} -lulmblas -Wl,-rpath,\${libdir}
Cflags: -I\${includedir}
EOF
echo "installed libulmblas in $PREFIX/lib"

if ! command -v "$FC" > /dev/null 2>&1; then
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
