#!/bin/sh
#
#  Runs the BLAS and CBLAS test programs of Reference-LAPACK 3.12.1 against
#  the libraries of ulmBLAS (make check-reference-lapack, see test/Makefile).
#  The sources are downloaded (and their checksum verified), not copied into
#  the repository; they are used unmodified.
#
#      reference-lapack.sh LIBDIR BUILDDIR
#
#  LIBDIR contains libcblas.a and libf77blas.a, BUILDDIR receives the
#  sources, the test programs and their output.  Environment: CC, FC, CXX
#  (linker, as for the Netlib suites of make check), CFLAGS, FFLAGS,
#  LDFLAGS, LDLIBS, FLIBS; LAPACK_TARBALL: use this file instead of
#  downloading it.
#
#  xGEMMTR (new in LAPACK 3.12, not part of ulmBLAS) is disabled in the input
#  files of the level 3 programs; stubs that abort satisfy the linker.  The
#  programs cblat1 and zblat1 of 3.12.1 report SCNRM2 and DZNRM2 as failed
#  for results that are correctly infinite (fixed in the development
#  version of LAPACK).  They are also run with the reference BLAS of the
#  tarball; the result is REF if all reported failures are exactly those of
#  the reference BLAS.
#
#  Exit status 0 if all programs report PASS or REF.
#
set -u

VERSION=3.12.1
URL=https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v$VERSION.tar.gz
SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422

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

cd "$BUILD" || exit 2
rm -rf run ref && mkdir run ref || exit 2

#
#  Support files: name mangling of the CBLAS test programs, xGEMMTR stubs
#
cat > cblas_mangling.h <<'EOF'
#ifndef F77_HEADER_INCLUDED
#define F77_HEADER_INCLUDED
#define F77_GLOBAL(lcname,UCNAME)  lcname##_
#endif
EOF
cat > gemmtr_stub.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
static void none(const char *f) { fprintf(stderr, "%s called\n", f); abort(); }
void sgemmtr_(void) { none("sgemmtr"); }
void dgemmtr_(void) { none("dgemmtr"); }
void cgemmtr_(void) { none("cgemmtr"); }
void zgemmtr_(void) { none("zgemmtr"); }
void cblas_sgemmtr(void) { none("cblas_sgemmtr"); }
void cblas_dgemmtr(void) { none("cblas_dgemmtr"); }
void cblas_cgemmtr(void) { none("cblas_cgemmtr"); }
void cblas_zgemmtr(void) { none("cblas_zgemmtr"); }
EOF
$CC $CFLAGS -c -o gemmtr_stub.o gemmtr_stub.c || exit 2

#  -fallow-argument-mismatch: needed by gfortran 10 or newer for some
#  programs (as for the Netlib suites of make check)
FFLAGS_TEST="$FFLAGS -w"
if echo | $FC -fallow-argument-mismatch -x f77 -c - -o /dev/null \
        > /dev/null 2>&1; then
    FFLAGS_TEST="$FFLAGS_TEST -fallow-argument-mismatch"
fi
CFLAGS_TEST="$CFLAGS -w -DADD_ -I$BUILD -I$SRC/CBLAS/include"

#
#  Reference BLAS of the tarball (only for the comparison of cblat1, zblat1)
#
mkdir -p refblas
for f in $SRC/BLAS/SRC/*.f $SRC/BLAS/SRC/*.f90; do
    o=refblas/`basename $f | sed 's/\.f9*0*$//'`.o
    [ -f $o ] || $FC $FFLAGS -w -c -o $o $f || exit 2
done
rm -f librefblas.a
ar cr librefblas.a refblas/*.o && ranlib librefblas.a || exit 2

#  link PROGRAM FORTRAN-SOURCE OBJECTS-AND-LIBRARIES...
link()
{
    prog=$1; src=$2; shift 2
    $FC $FFLAGS_TEST -c -o $prog.o $src \
    && $CXX $LDFLAGS -o $prog $prog.o "$@" $LDLIBS $FLIBS
}

#  verdict LEVEL OUTPUT STATUS [REFERENCE-OUTPUT]
verdict()
{
    if [ $3 -ne 0 ] || [ ! -s $2 ]; then echo FAIL; return; fi
    if [ $1 != 1 ] && ! grep -q 'END OF TESTS' $2; then echo FAIL; return; fi
    pat='[*]{7}|SUSPECT|FAIL|NOT RECOGNIZED|test='
    if ! grep -qE "$pat" $2; then echo PASS; return; fi
    if [ $# -ge 4 ] && [ -s "$4" ] \
       && [ "`grep -E "$pat" $2`" = "`grep -E "$pat" $4`" ]; then
        echo REF; return
    fi
    echo FAIL
}

status=0
: > result.txt
for t in s d c z; do
    T=`echo $t | tr 'a-z' 'A-Z'`
    for l in 1 2 3; do
        # Fortran 77 interface
        p=${t}blat$l
        if ! link run/$p $SRC/BLAS/TESTING/$p.f $LIBDIR/libf77blas.a \
                gemmtr_stub.o > run/$p.build 2>&1; then
            echo "FAIL  F77 $p (build, see $BUILD/run/$p.build)" \
                | tee -a result.txt
            status=1; continue
        fi
        ref=
        if [ $l = 1 ]; then
            (cd run && ./$p > $p.out 2>&1); rc=$?
            out=run/$p.out
            if [ $t = c ] || [ $t = z ]; then
                link ref/$p $SRC/BLAS/TESTING/$p.f librefblas.a \
                    > ref/$p.build 2>&1 && (cd ref && ./$p > $p.out 2>&1)
                ref=ref/$p.out
            fi
        else
            sed "s/^\(${T}GEMMTR *\)T/\1F/" $SRC/BLAS/TESTING/$p.in \
                > run/$p.in
            (cd run && ./$p < $p.in > $p.log 2>&1); rc=$?
            out=run/$p.out
        fi
        r=`verdict $l $out $rc $ref`
        [ $r = FAIL ] && status=1
        echo "`printf %-5s $r` F77 $p (see $BUILD/$out)" | tee -a result.txt
    done
    # C interface
    for f in c_${t}blas1 c_${t}blas2 c_${t}blas3 c_${t}2chke c_${t}3chke \
             auxiliary c_xerbla; do
        [ -f run/$f.o ] \
            || $CC $CFLAGS_TEST -c -o run/$f.o $SRC/CBLAS/testing/$f.c \
            || exit 2
    done
    for l in 1 2 3; do
        p=c_${t}blat$l
        objs=run/c_${t}blas$l.o
        [ $l != 1 ] && objs="$objs run/c_${t}${l}chke.o run/auxiliary.o run/c_xerbla.o"
        if ! link run/$p $SRC/CBLAS/testing/$p.f $objs $LIBDIR/libcblas.a \
                $LIBDIR/libf77blas.a gemmtr_stub.o > run/$p.build 2>&1; then
            echo "FAIL  C   $p (build, see $BUILD/run/$p.build)" \
                | tee -a result.txt
            status=1; continue
        fi
        if [ $l = 1 ]; then
            (cd run && ./$p > $p.out 2>&1); rc=$?
        else
            sed "s/^\(cblas_${t}gemmtr *\)T/\1F/" \
                $SRC/CBLAS/testing/${t}in$l > run/${t}in$l
            (cd run && ./$p < ${t}in$l > $p.out 2>&1); rc=$?
        fi
        r=`verdict $l run/$p.out $rc`
        [ $r = FAIL ] && status=1
        echo "`printf %-5s $r` C   $p (see $BUILD/run/$p.out)" | tee -a result.txt
    done
done

echo
echo "Reference-LAPACK $VERSION BLAS test programs:" \
     "`grep -c '^PASS' result.txt` PASS," \
     "`grep -c '^REF' result.txt` REF (failures identical to the reference BLAS)," \
     "`grep -c '^FAIL' result.txt` FAIL"
exit $status
