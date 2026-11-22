#! /bin/sh
#
#  flexiblas-test.sh [64]: ulmBLAS (installed, with the FlexiBLAS
#  configuration of packaging/fedora) as the backend of FlexiBLAS: builds
#  and runs flexiblas-test.c with FLEXIBLAS=ULMBLAS, and checks with
#  LD_DEBUG that FlexiBLAS binds dgemm_ and the LAPACK functions of ulmBLAS
#  (dgetrf_, dpotrf_) to libulmblas.  With 64: FlexiBLAS with 64-bit
#  integers (libflexiblas64, FLEXIBLAS64=ULMBLAS) and libulmblas64.  Needs
#  cc, pkg-config and FlexiBLAS (flexiblas, flexiblas-devel).
#
set -u
HERE=`dirname "$0"`
if [ "${1:-}" = 64 ]; then
    FB=flexiblas64; NAME=libulmblas64; LIB='libulmblas64\.'; DEFS=-DFLEXIBLAS64
else
    FB=flexiblas; NAME=libulmblas; LIB='libulmblas\.'; DEFS=
fi
EXE=${TMPDIR:-/tmp}/$FB-test
LOG=${TMPDIR:-/tmp}/$FB-test.log

$FB list
cc -O2 $DEFS -o "$EXE" "$HERE/flexiblas-test.c" `pkg-config --cflags --libs $FB` \
    -ldl || exit 1
if [ "$FB" = flexiblas64 ]; then
    FLEXIBLAS64=ULMBLAS; FLEXIBLAS64_VERBOSE=1; export FLEXIBLAS64 FLEXIBLAS64_VERBOSE
else
    FLEXIBLAS=ULMBLAS; FLEXIBLAS_VERBOSE=1; export FLEXIBLAS FLEXIBLAS_VERBOSE
fi
"$EXE" 2>&1 | tail -30
unset FLEXIBLAS_VERBOSE FLEXIBLAS64_VERBOSE
LD_DEBUG=bindings LD_DEBUG_OUTPUT=$LOG "$EXE" > /dev/null 2>&1
status=0
#  "binding file ... to .../libulmblas.so.0 [0]: normal symbol `dgetrf_'"
#  (the lookup of FlexiBLAS in the backend)
for f in dgemm_ dgetrf_ dpotrf_; do
    if grep -h "normal symbol .$f'" "$LOG".* | grep -q "to [^ ]*$LIB"; then
        echo "$f: $NAME"
    else
        echo "$f: not from $NAME: FAIL"
        grep -h "normal symbol .$f'" "$LOG".*
        status=1
    fi
done
/bin/rm -f "$LOG".*
"$EXE" || status=1
exit $status
