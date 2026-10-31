#!/bin/sh
#
#  Differential conformance test of the BLAS interfaces (make
#  check-conformance, see test/blas_conformance.md): the programs level1.c
#  and level23.cc call every level 1, 2 and 3 routine of the Fortran 77 and
#  C interfaces with many combinations of sizes, increments, leading
#  dimensions, flags and special values, with guard cells around all
#  operands, and print the results and the complete buffers.  Each program
#  is linked once with ulmBLAS and once with the reference BLAS and CBLAS of
#  LAPACK 3.12.1 (built from the sources of the tarball that
#  reference-lapack.sh downloads); compare.cc compares the outputs.
#
#      conformance.sh LIBDIR BUILDDIR
#
#  LIBDIR contains libcblas.a, libf77blas.a, cblas.h and f77blas.h of an
#  ulmBLAS build with index type int, BUILDDIR receives the sources and the
#  programs (the same directory as for reference-lapack.sh, so the tarball
#  is downloaded only once).  Environment: CC, FC, CXX, CFLAGS, FFLAGS,
#  LDFLAGS, LDLIBS, FLIBS, LAPACK_TARBALL as for reference-lapack.sh.
#  expected.txt lists the documented differences (test/known_issues.md).
#
#  Exit status 0 if all calls agree (apart from the expected differences).
#
set -u

VERSION=3.12.1
URL=https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v$VERSION.tar.gz
SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422

HERE=$(cd "$(dirname "$0")" && pwd) || exit 2
LIBDIR=$(cd "$1" && pwd) || exit 2
BUILD=$2
CC=${CC:-cc}
FC=${FC:-gfortran}
CXX=${CXX:-c++}
CFLAGS=${CFLAGS:-}
FFLAGS=${FFLAGS:-}
LDFLAGS=${LDFLAGS:-}
LDLIBS=${LDLIBS:-}
FLIBS=${FLIBS:-}

mkdir -p "$BUILD" && BUILD=$(cd "$BUILD" && pwd) || exit 2
SRC=$BUILD/lapack-$VERSION
TARBALL=${LAPACK_TARBALL:-$BUILD/lapack-$VERSION.tar.gz}

sha256()
{
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

if [ ! -d "$SRC" ]; then
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
        lapack-$VERSION/BLAS lapack-$VERSION/CBLAS lapack-$VERSION/LICENSE) \
        || exit 2
fi

WORK=$BUILD/conformance
mkdir -p $WORK/ref && cd $WORK || exit 2

#
#  Reference BLAS and CBLAS
#
cat > cblas_mangling.h <<'EOT'
#ifndef F77_HEADER_INCLUDED
#define F77_HEADER_INCLUDED
#define F77_GLOBAL(lcname,UCNAME)  lcname##_
#endif
EOT
for f in $SRC/BLAS/SRC/*.f $SRC/BLAS/SRC/*.f90 $SRC/CBLAS/src/*.f; do
    o=ref/`basename $f | sed 's/\.f9*0*$//'`.o
    [ -f $o ] || $FC $FFLAGS -w -c -o $o $f || exit 2
done
for f in $SRC/CBLAS/src/*.c; do
    o=ref/`basename $f .c`.o
    [ -f $o ] || $CC $CFLAGS -w -DADD_ -I$WORK -I$SRC/CBLAS/include \
                     -c -o $o $f || exit 2
done
rm -f libref.a
ar cr libref.a ref/*.o && ranlib libref.a || exit 2

#
#  Test programs, linked twice
#
$CC $CFLAGS -c -o level1.o $HERE/level1.c || exit 2
$CXX -std=c++17 -I$LIBDIR -c -o level23.o $HERE/level23.cc || exit 2
$CXX -o compare $HERE/compare.cc || exit 2
for p in level1 level23; do
    $CXX $LDFLAGS -o $p-ulmblas $p.o $LIBDIR/libcblas.a $LIBDIR/libf77blas.a \
        $LDLIBS $FLIBS || exit 2
    $CXX $LDFLAGS -o $p-reference $p.o libref.a $LDLIBS $FLIBS || exit 2
done

status=0
run()
{
    name=$1; shift
    ./$name-ulmblas "$@" > $name$*-ulmblas.out 2>&1 &
    ./$name-reference "$@" > $name$*-reference.out 2>&1
    r=$?
    wait $! && [ $r -eq 0 ] || { echo "FAIL  $name $*: program failed"; status=1; return; }
    if ./compare $name$*-ulmblas.out $name$*-reference.out \
                 $HERE/expected.txt > $name$*.cmp; then
        echo "PASS  $name $* (`head -1 $name$*.cmp`)"
    else
        echo "FAIL  $name $* (see $WORK/$name$*.cmp)"
        status=1
    fi
}
run level1
for g in gemv gbmv symv ger syr trv gemm symm syrk trm; do
    run level23 $g
done
exit $status
