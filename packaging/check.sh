#! /bin/sh
#
#  check.sh: the test of a package build (the check step of the recipes in
#  packaging/), run after interfaces/install-shared.sh has built the
#  libraries in this source tree (ULMBLAS_STEP=build).  The Netlib test
#  programs of the Fortran 77 interface and of CBLAS level 1
#  (test/check-shared.sh) against libulmblas of this tree; with the
#  dispatching library (x86-64, Apple Silicon) once with the backend it
#  selects for the CPU and once with the portable one (generic on x86-64).
#  Needs FC (default gfortran) and CC (default cc); no network.
#
set -u
TOP=$(cd "$(dirname "$0")/.." && pwd)
cd "$TOP" || exit 1
H=$(ls interfaces/blas/dispatch/build.dispatch/*/cblas.h \
       interfaces/blas/shared/build.shared/cblas.h 2> /dev/null | head -1)
[ -n "$H" ] || { echo "check.sh: build first (install-shared.sh)" >&2; exit 2; }
INC=$(dirname "$H")
#  the backends of this tree, not those of an installed package (the
#  dispatching library may have the directory of the package compiled in)
ULMBLAS_BACKEND_DIR=$TOP
export ULMBLAS_BACKEND_DIR

status=0
sh test/check-shared.sh . "$INC" build/check ulmblas f77 || status=1
if [ -f libulmblas_generic.so ] || [ -f libulmblas_generic.dll ]; then
    ULMBLAS_BACKEND=generic sh test/check-shared.sh . "$INC" \
        build/check-generic ulmblas f77 || status=1
fi
exit $status
