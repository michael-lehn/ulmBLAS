#! /bin/sh
#
#  build.sh LIBDIR BUILDDIR [test|test-all]: Reference-LAPACK 3.12.1
#  (unmodified, see fetch.sh) as a shared library libulmlapack.so /
#  libulmlapack.dylib
#  (Windows: libulmlapack.dll) in LIBDIR, built against the shared library
#  libulmblas in LIBDIR (make shared or make shared-dispatch) instead of the
#  reference BLAS of the tarball.  BUILDDIR receives the sources and the
#  build.  With "test" or "test-all" the test programs of LAPACK (linear
#  equations and eigenvalue problems, TESTING) are built against its
#  archive and libulmblas and run (pass criterion below).
#
#  The library has its own name, so that it cannot be confused with a
#  liblapack of the system; it contains the LAPACK routines, LAPACK's own
#  XERBLA and its auxiliary routines, no BLAS.  Built with the makefiles of
#  LAPACK (GNU make; MAKE, default make) and a Fortran compiler (FC, default
#  gfortran).  Flags: FFLAGS, then -frecursive (local arrays on the stack:
#  thread safety), -fPIC (not on Windows) and LAPACK_FOPT (default -O2, as
#  LAPACK's CMake build, which reduces -O3 to -O2: with gfortran -O3 the
#  test programs of the eigenvalue drivers report about 200 results above
#  the threshold instead of 3).
#  The files that LAPACK's makefiles compile without optimization
#  (?laruv, ?la_wwaddw, the matrix generators ?laran) get -O0 instead of
#  LAPACK_FOPT.  LAPACK_TIMER: the timer of LAPACK's make.inc (default
#  INT_ETIME).  JOBS: parallel jobs of make (default: the number of CPUs).
#  The test programs need python3 (lapack_testing.py).  The
#  sources are downloaded unless LAPACK_TARBALL names the tarball
#  (fetch.sh), e.g. for builds without network access.
#
set -u
LIBDIR=$(cd "$1" && pwd) || exit 2
mkdir -p "$2" && BUILD=$(cd "$2" && pwd) || exit 2
TEST=${3:-}
case $TEST in
    ''|test|test-all) ;;
    *) echo "usage: build.sh LIBDIR BUILDDIR [test|test-all]" >&2; exit 2 ;;
esac
FC=${FC:-gfortran}
MAKECMD=${MAKE:-make}
VERSION=3.12.1
HERE=$(cd "`dirname "$0"`" && pwd)
SRC=$BUILD/lapack-$VERSION

case `uname -s` in
    Darwin) SO=dylib; BLAS=$LIBDIR/libulmblas.dylib; PIC=-fPIC ;;
    #  Windows (MinGW-w64): libulmlapack.dll with the import libraries
    #  libulmlapack.dll.a (MinGW) and ulmlapack.lib (MSVC), linked against
    #  the import library of libulmblas.dll
    MINGW*|MSYS*) SO=dll; BLAS=$LIBDIR/libulmblas.dll.a; PIC= ;;
    *)      SO=so; BLAS=$LIBDIR/libulmblas.so; PIC=-fPIC ;;
esac
[ -f "$BLAS" ] || { echo "error: $BLAS not found (make shared)" >&2; exit 2; }

sh "$HERE/fetch.sh" "$BUILD" || exit 2

#
#  make.inc of LAPACK (see make.inc.example of the tarball).  If it differs
#  from the one of an earlier build, the objects of that build are removed.
#
FFLAGS_ALL="${FFLAGS:-} -frecursive $PIC"
cat > "$BUILD/make.inc" <<EOF
SHELL = /bin/sh
FC = $FC
FFLAGS = $FFLAGS_ALL ${LAPACK_FOPT:--O2}
FFLAGS_DRV = \$(FFLAGS)
FFLAGS_NOOPT = $FFLAGS_ALL -O0
LDFLAGS =
AR = ${AR:-ar}
ARFLAGS = cr
RANLIB = ${RANLIB:-ranlib}
TIMER = ${LAPACK_TIMER:-INT_ETIME}
BLASLIB = $BLAS
LAPACKLIB = \$(TOPSRCDIR)/liblapack.a
TMGLIB = \$(TOPSRCDIR)/libtmglib.a
EOF
if ! cmp -s "$BUILD/make.inc" "$SRC/make.inc"; then
    find "$SRC" \( -name '*.o' -o -name '*.mod' -o -name '*.a' \) \
        -exec rm -f {} + || exit 1
    cp "$BUILD/make.inc" "$SRC/make.inc" || exit 1
fi

JOBS=${JOBS:-`getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4`}
$MAKECMD -j "$JOBS" -C "$SRC/SRC" > "$BUILD/build.log" 2>&1 \
    || { tail -20 "$BUILD/build.log"; echo "error: build failed" >&2; exit 1; }

[ -f "$SRC/liblapack.a" ] || { echo "error: $SRC/liblapack.a not built" >&2; exit 1; }

#
#  The LAPACK routines that libulmblas exports itself (LAPACK_EXPORTS in
#  interfaces/blas/shared/Makefile: ?getrf, ?potrf, ... of ulmBLAS) are
#  left out: their objects are removed from a copy of the archive, so
#  libulmlapack, the callers inside it (?gesvx, ?gecon, ...) and the test
#  programs use those of libulmblas.  The routines that both
#  provide for the error handling (xerbla, xerbla_array, lsame) stay.
#
case $SO in
    dylib) EXPORTED=`${NM:-nm} -gU "$BLAS" | awk 'NF == 3 && $2 == "T" {print $3}' | sed 's/^_//'` ;;
    dll)   EXPORTED=`${NM:-nm} -g "$BLAS" | awk 'NF == 3 && $2 == "T" {print $3}'` ;;
    *)     EXPORTED=`${NM:-nm} -D --defined-only "$BLAS" | awk 'NF == 3 && $2 == "T" {print $3}'` ;;
esac
[ -n "$EXPORTED" ] || { echo "error: no symbols in $BLAS" >&2; exit 1; }
OURS=`for s in $EXPORTED; do echo $s; done | sed -n 's/^\([a-z][a-z0-9]*\)_$/\1/p' \
      | grep -v -E '^(xerbla|xerbla_array|lsame)$' | sort -u`
${AR:-ar} t "$SRC/liblapack.a" | sed -n 's/\.o$//p' | sort -u > "$BUILD/members.txt"
LEFT_OUT=`printf '%s\n' "$OURS" | comm -12 - "$BUILD/members.txt"`
ARCHIVE=$BUILD/libulmlapack.a
cp "$SRC/liblapack.a" "$ARCHIVE" || exit 1
if [ -n "$LEFT_OUT" ]; then
    ${AR:-ar} d "$ARCHIVE" `for r in $LEFT_OUT; do echo $r.o; done` || exit 1
    ${RANLIB:-ranlib} "$ARCHIVE" || exit 1
fi
echo "from libulmblas:" $LEFT_OUT
OUT=$LIBDIR/libulmlapack.$SO
if [ "$SO" = dll ]; then
    #  the Fortran runtime linked statically (and not exported): the DLL
    #  imports only libulmblas.dll and DLLs of the system
    $FC -shared -Wl,--whole-archive "$ARCHIVE" -Wl,--no-whole-archive \
        -Wl,--export-all-symbols \
        -Wl,--exclude-libs,libgfortran.a:libquadmath.a:libgcc.a:libgcc_eh.a:libwinpthread.a:libmingwex.a \
        -Wl,--out-implib,"$LIBDIR/libulmlapack.dll.a" \
        -Wl,--output-def,"$BUILD/libulmlapack.def" \
        -L"$LIBDIR" -lulmblas -static -static-libgfortran -static-libgcc \
        -Wl,--no-undefined -o "$OUT" || exit 1
    ${DLLTOOL:-dlltool} -d "$BUILD/libulmlapack.def" -D libulmlapack.dll \
        -l "$LIBDIR/ulmlapack.lib" || exit 1
elif [ "$SO" = dylib ]; then
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

#
#  The test programs of TESTING (LIN, EIG) against the archive of
#  libulmlapack and libulmblas (with the shared library, LAPACK's routines
#  would call its own XERBLA on macOS and Windows, not the one of the test
#  programs that checks the error exits).
#
#  Pass criterion: every program ran to its end ("End of tests", the DMD
#  programs "Test completed"), no error exit failed ("Illegal" in the
#  output) and there are no numerical errors (lapack_testing.py: results
#  above the threshold, routines that returned INFO != 0) except, with
#  "test", the known ones: known-failures.txt lists the borderline cases
#  that flip with the rounding of the BLAS on one of the machines tested
#  (failures.sh; with the reference BLAS of the tarball there are about 150
#  on Apple M5).  "test-all" fails on these too.  The BLAS itself is
#  checked by the ATLAS testers (bench/).  Thresholds: those of LAPACK,
#  except where OpenBLAS uses a higher one and ulmBLAS fails LAPACK's:
#  STFSM (stest_rfp.in) 45 instead of 30.
#
if [ -n "$TEST" ]; then
    LAPACKLIB=$ARCHIVE
    #  TESTLDFLAGS, Windows: the programs export their XERBLA, which the
    #  LAPACK functions in libulmblas.dll call then (a DLL calls its own
    #  xerbla_ otherwise, see interfaces/blas/F77/xerbla.cc)
    case $SO in
        dll) TESTLDFLAGS=-Wl,--export-all-symbols ;;
        *)   TESTLDFLAGS=-Wl,-rpath,$LIBDIR ;;
    esac
    command -v python3 > /dev/null 2>&1 \
        || { echo "error: the LAPACK tests need python3" >&2; exit 2; }
    #  the programs first, one make per directory (the programs of a
    #  directory share objects), then the runs in parallel
    rm -f "$SRC"/TESTING/*.out
    sed 's/^30\.0 /45.0 /' "$SRC/TESTING/stest_rfp.in" > "$BUILD/stest_rfp.in" \
        && cp "$BUILD/stest_rfp.in" "$SRC/TESTING/stest_rfp.in" || exit 1
    LIBS="LAPACKLIB=$LAPACKLIB"
    { $MAKECMD -j "$JOBS" -C "$SRC/TESTING/MATGEN" \
        && $MAKECMD -C "$SRC/TESTING/LIN" cleanexe \
        && $MAKECMD -C "$SRC/TESTING/EIG" cleanexe \
        && $MAKECMD -j "$JOBS" -C "$SRC/TESTING/LIN" "$LIBS" \
                    "BLASLIB=$BLAS" "LDFLAGS=$TESTLDFLAGS" \
        && $MAKECMD -j "$JOBS" -C "$SRC/TESTING/EIG" "$LIBS" \
                    "BLASLIB=$BLAS" "LDFLAGS=$TESTLDFLAGS" \
        && $MAKECMD -j "$JOBS" -C "$SRC/TESTING" "$LIBS" \
                    "BLASLIB=$BLAS" "LDFLAGS=$TESTLDFLAGS"; } \
        > "$BUILD/test.log" 2>&1 \
        || { tail -20 "$BUILD/test.log"; echo "error: tests failed" >&2; exit 1; }
    (cd "$SRC" && python3 lapack_testing.py > "$BUILD/summary.log" 2>&1)
    grep -A8 -E "^ *SUMMARY" "$BUILD/summary.log"
    grep -E "out of|INFO|[Ii]llegal" "$BUILD/summary.log" | sort -u
    #  (Python on Windows ends the lines with CR LF)
    errors=`cd "$SRC" && python3 lapack_testing.py -n | tail -2 | tr -d '\r' | tr '\n' ' '`
    set -- $errors
    unfinished=
    for f in "$SRC"/TESTING/*.out; do
        case $f in
            *dmd.out) grep -q "Test completed" "$f" ;;
            *)        grep -q "End of tests" "$f" ;;
        esac || unfinished="$unfinished `basename "$f"`"
    done
    illegal=`cat "$SRC"/TESTING/*.out | grep -c "[Ii]llegal"`
    numerical=$((${1:-0} + ${2:-0} - illegal))
    [ $# = 2 ] || { echo "error: lapack_testing.py -n: $errors" >&2; exit 1; }
    sh "$HERE/failures.sh" "$SRC/TESTING" "$HERE/known-failures.txt" \
        > "$BUILD/failures.log" || exit 1
    grep -v '^known failures:' "$BUILD/failures.log" | sort | uniq -c
    known=`tail -1 "$BUILD/failures.log" | awk '{ print $3 }'`
    new=$((numerical - known))
    echo "LAPACK tests: $numerical numerical errors ($known known, $new new)," \
         "$illegal failed error exits, unfinished:${unfinished:- none}" \
         "(see $BUILD/summary.log, $BUILD/failures.log)"
    #  more known failures than numerical errors: failures.sh misread the
    #  output
    [ "$new" -ge 0 ] || { echo "error: failures.sh does not match lapack_testing.py" >&2; exit 1; }
    [ -z "$unfinished" ] && [ "$illegal" = 0 ] || exit 1
    case $TEST in
        test)     [ "$new" = 0 ] ;;
        test-all) [ "$numerical" = 0 ] ;;
    esac
    exit $?
fi
