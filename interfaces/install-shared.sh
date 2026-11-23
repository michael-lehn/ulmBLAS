#! /bin/sh
#
#  install-shared.sh PREFIX: builds the shared libraries of ulmBLAS and
#  installs them for other programs (e.g. NumPy, interfaces/python/numpy.sh):
#  runs make install-shared PREFIX=PREFIX in the source tree (one make; the
#  files installed and the settings: mk/install.mk).  For the recipes of
#  packages that call this script; make install-shared does the same.
#
#      PREFIX/lib/libulmblas.0.dylib (.so.0, Windows: bin/libulmblas.dll
#                                          and lib/libulmblas.dll.a) and on
#                                          Apple Silicon and x86-64 its
#                                          backends
#      PREFIX/lib/libulmlapack.dylib (.so) Reference-LAPACK 3.12.1 on top of
#                                          libulmblas (needs a Fortran
#                                          compiler, FC)
#      PREFIX/lib/libulmblas64.0.dylib (.so.0; not on Windows)
#      PREFIX/include/cblas.h, PREFIX/include/ulmblas64/cblas.h
#      PREFIX/lib/pkgconfig/ulmblas.pc, ulmblas64.pc, ulmlapack.pc
#
#  Run from anywhere; the build happens in the source tree with GNU make
#  (MAKE, default gmake if found, else make; it may contain options, e.g.
#  MAKE="make -j8").
#
#  Environment (for packages; all optional; see mk/install.mk): DESTDIR,
#  ULMBLAS_LIBDIR, ULMBLAS_BINDIR, ULMBLAS_INCLUDEDIR, ULMBLAS_RPATH,
#  ULMBLAS_VERSION, ULMBLAS_STATIC_LIBSTDCXX, ULMBLAS_BACKENDDIR,
#  ULMBLAS_SONAMES, ULMBLAS_SONAMEDIR, ULMBLAS_INTERFACE64, FC, and
#
#      ULMBLAS_STEP        build: only build (make build-shared), install
#                          nothing (the build step of a package; its install
#                          step runs the script again with the same
#                          environment)
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
MAKECMD=${MAKE:-}
unset MAKE
if [ -z "$MAKECMD" ]; then
    if command -v gmake > /dev/null 2>&1; then MAKECMD=gmake; else MAKECMD=make; fi
fi
TARGET=install-shared
if [ "${ULMBLAS_STEP:-}" = build ]; then
    TARGET=build-shared
fi
#  build tools may have set variables for themselves that are also make
#  variables of ulmBLAS (rattler-build: BUILD_DIR)
unset BUILD_DIR OUTDIR
exec $MAKECMD -C "$TOP" $TARGET PREFIX="$PREFIX"
