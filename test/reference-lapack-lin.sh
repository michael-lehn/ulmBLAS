#!/bin/sh
#
#  Runs the test programs of Reference-LAPACK 3.12.1 for linear equations
#  (TESTING/LIN, xlintst[sdcz]) on the paths GE, PO and TR with the LAPACK
#  functions of ulmBLAS (make check-reference-lapack-lin, see
#  test/Makefile).  The sources are downloaded (and their checksum
#  verified), not copied into the repository; they are used unmodified.
#
#      reference-lapack-lin.sh LIBDIR BUILDDIR
#
#  LIBDIR contains libf77lapack.a (the LAPACK functions of ulmBLAS) and
#  libf77blas.a.  Reference-LAPACK is built from the tarball, the objects
#  of the functions ulmBLAS has (ROUTINES below) are removed from its
#  library, so the test programs and all other LAPACK routines call those
#  of ulmBLAS; BLAS is ulmBLAS too.  The input files are those of the
#  tarball reduced to the paths GE, PO and TR (routines, drivers and error
#  exits).
#  Note: the programs use matrices with at most 132 rows, so ulmBLAS
#  factors them with its serial algorithms.
#
#  Environment: FC, FFLAGS, LDLIBS (e.g. the C++ runtime and OpenMP);
#  LAPACK_TARBALL: use this file instead of downloading it.
#
#  Exit status 0 if all paths pass.
#
set -u
VERSION=3.12.1
URL=https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v$VERSION.tar.gz
SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422
LIBDIR=$(cd "$1" && pwd) || exit 2
BUILD=$2
FC=${FC:-gfortran}
FFLAGS=${FFLAGS:--O2}
LDLIBS=${LDLIBS:-}
if [ -z "$LDLIBS" ]; then
    if [ "`uname -s`" = Darwin ]; then LDLIBS=-lc++; else LDLIBS=-lstdc++; fi
fi
JOBS=${JOBS:-4}

mkdir -p "$BUILD" && BUILD=$(cd "$BUILD" && pwd) || exit 2
SRC=$BUILD/lapack-$VERSION
TARBALL=${LAPACK_TARBALL:-$BUILD/lapack-$VERSION.tar.gz}
case $TARBALL in /*) ;; *) TARBALL=`pwd`/$TARBALL ;; esac

sha256()
{
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

if [ ! -d "$SRC/TESTING/LIN" ]; then
    if [ ! -f "$TARBALL" ]; then
        echo "downloading $URL"
        if command -v curl > /dev/null 2>&1; then
            curl -fsSL -o "$TARBALL.tmp" "$URL"
        else
            wget -q -O "$TARBALL.tmp" "$URL"
        fi || { echo "error: download failed" >&2; exit 2; }
        mv "$TARBALL.tmp" "$TARBALL"
    fi
    if [ "`sha256 "$TARBALL"`" != "$SHA256" ]; then
        echo "error: checksum of $TARBALL does not match" >&2
        exit 2
    fi
    (cd "$BUILD" && gzip -dc "$TARBALL" | tar -xf - \
        lapack-$VERSION/SRC lapack-$VERSION/INSTALL lapack-$VERSION/TESTING \
        lapack-$VERSION/LICENSE) || exit 2
fi
cd "$SRC" || exit 2

#
#  Reference-LAPACK and its matrix generators; then a copy of the library
#  without ?GETRF and ?POTRF
#
sed -e "s|^FC = .*|FC = $FC|" -e "s|^FFLAGS = .*|FFLAGS = $FFLAGS -frecursive|" \
    -e "s|^FFLAGS_NOOPT = .*|FFLAGS_NOOPT = -O0 -frecursive|" \
    INSTALL/make.inc.gfortran > make.inc || exit 2
echo "building Reference-LAPACK $VERSION (SRC, TESTING/MATGEN)"
make -j$JOBS -C SRC > $BUILD/lapack.log 2>&1 \
    && make -j$JOBS -C TESTING/MATGEN >> $BUILD/lapack.log 2>&1 \
    || { echo "error: see $BUILD/lapack.log" >&2; exit 2; }
ROUTINES="getrf getrs gesv getri potrf potrs posv potri trtrs trtri lauum laswp"
OBJECTS=`for r in $ROUTINES; do for t in s d c z; do printf '%s ' $t$r.o; done; done`
cp liblapack.a liblapack_ulmblas.a \
    && ar d liblapack_ulmblas.a $OBJECTS \
    && ranlib liblapack_ulmblas.a || exit 2

echo "building xlintst[sdcz] with the LAPACK functions and BLAS of ulmBLAS"
rm -f TESTING/LIN/xlintst?
make -j$JOBS -C TESTING/LIN LAPACKLIB=$SRC/liblapack_ulmblas.a \
     "BLASLIB=$LIBDIR/libf77lapack.a $LIBDIR/libf77blas.a" \
     "LDFLAGS=$LDLIBS" xlintsts xlintstd xlintstc xlintstz \
     > $BUILD/lin.log 2>&1 || { echo "error: see $BUILD/lin.log" >&2; exit 2; }

status=0
: > $BUILD/result.txt
for t in s d c z; do
    T=`echo $t | tr 'a-z' 'A-Z'`
    awk -v P=$T 'NR<=16 { print; next } $1==P"GE" || $1==P"PO" || $1==P"TR" { print }' \
        TESTING/${t}test.in > $BUILD/${t}test.in
    (cd TESTING/LIN && ./xlintst$t < $BUILD/${t}test.in > $BUILD/xlintst$t.out 2>&1)
    out=$BUILD/xlintst$t.out
    passed=`grep -cE 'All tests for .* passed the threshold|passed the tests of the error exits' $out`
    if [ $passed -eq 10 ] && grep -q 'End of tests' $out; then
        r=PASS
    else
        r=FAIL; status=1
    fi
    runs=`grep -E 'All tests for' $out | sed 's/.*( *\([0-9]*\) tests run).*/\1/' \
          | awk '{s+=$1} END {print s+0}'`
    echo "`printf %-5s $r` xlintst$t GE, PO, TR ($runs tests; see $out)" \
        | tee -a $BUILD/result.txt
done
echo
echo "Reference-LAPACK $VERSION LIN test programs (GE, PO, TR):" \
     "`grep -c '^PASS' $BUILD/result.txt` PASS, `grep -c '^FAIL' $BUILD/result.txt` FAIL"
exit $status
