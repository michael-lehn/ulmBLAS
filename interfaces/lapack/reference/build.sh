#! /bin/sh
#
#  build.sh LIBDIR BUILDDIR [test]: Reference-LAPACK 3.12.1 (unmodified,
#  see fetch.sh) as a shared library libulmlapack.so / libulmlapack.dylib
#  in LIBDIR, built against the shared library libulmblas in LIBDIR (make
#  shared or make shared-dispatch) instead of the reference BLAS of the
#  tarball.  BUILDDIR receives the sources and the CMake build.  With
#  "test" the test programs of LAPACK (linear equations and eigenvalue
#  problems, ctest) are built and run as well, against libulmlapack and
#  libulmblas.
#
#  The library has its own name, so that it cannot be confused with a
#  liblapack of the system; it contains the LAPACK routines, LAPACK's own
#  XERBLA and its auxiliary routines, no BLAS.  Needs CMake and a Fortran
#  compiler (FC, default gfortran); FFLAGS are passed on.
#
set -u
LIBDIR=$(cd "$1" && pwd) || exit 2
mkdir -p "$2" && BUILD=$(cd "$2" && pwd) || exit 2
TEST=${3:-}
FC=${FC:-gfortran}
VERSION=3.12.1
HERE=$(cd "`dirname "$0"`" && pwd)

case `uname -s` in
    Darwin) SO=dylib ;;
    *)      SO=so ;;
esac
BLAS=$LIBDIR/libulmblas.$SO
[ -f "$BLAS" ] || { echo "error: $BLAS not found (make shared)" >&2; exit 2; }

sh "$HERE/fetch.sh" "$BUILD" || exit 2

if [ "$TEST" = test ]; then TESTING=ON; else TESTING=OFF; fi

#  CMake builds the static library of LAPACK (position independent code);
#  the shared library is linked from it below, so that its name, install
#  name (soname) and run path are those of ulmBLAS.  USE_OPTIMIZED_BLAS:
#  link the test programs against BLAS_LIBRARIES, do not build the
#  reference BLAS.  BUILD_INDEX64_EXT_API=OFF: no second interface with
#  64-bit integers (its routines would call dgemm_64_ etc.).
cmake -S "$BUILD/lapack-$VERSION" -B "$BUILD/cmake" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_Fortran_COMPILER="$FC" \
      -DCMAKE_Fortran_FLAGS="${FFLAGS:-}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DUSE_OPTIMIZED_BLAS=ON \
      -DBLAS_LIBRARIES="$BLAS" \
      -DBUILD_INDEX64_EXT_API=OFF \
      -DCBLAS=OFF -DLAPACKE=OFF \
      -DBUILD_TESTING=$TESTING \
      > "$BUILD/cmake.log" 2>&1 \
    || { tail -20 "$BUILD/cmake.log"; echo "error: cmake failed" >&2; exit 1; }

JOBS=`getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4`
cmake --build "$BUILD/cmake" -j "$JOBS" > "$BUILD/build.log" 2>&1 \
    || { tail -20 "$BUILD/build.log"; echo "error: build failed" >&2; exit 1; }

ARCHIVE=$BUILD/cmake/lib/liblapack.a
[ -f "$ARCHIVE" ] || { echo "error: $ARCHIVE not built" >&2; exit 1; }
OUT=$LIBDIR/libulmlapack.$SO
if [ "$SO" = dylib ]; then
    $FC -dynamiclib -install_name @rpath/libulmlapack.dylib \
        -Wl,-force_load,"$ARCHIVE" -L"$LIBDIR" -lulmblas \
        -Wl,-rpath,@loader_path -o "$OUT" || exit 1
else
    $FC -shared -Wl,-soname,libulmlapack.so \
        -Wl,--whole-archive "$ARCHIVE" -Wl,--no-whole-archive \
        -L"$LIBDIR" -lulmblas -Wl,-rpath,'$ORIGIN' -Wl,--no-undefined \
        -o "$OUT" || exit 1
fi
echo "built $OUT"

if [ "$TEST" = test ]; then
    (cd "$BUILD/cmake" && ctest -j "$JOBS" --output-on-failure \
        > "$BUILD/ctest.log" 2>&1)
    status=$?
    grep -E "tests passed|tests failed" "$BUILD/ctest.log"
    exit $status
fi
