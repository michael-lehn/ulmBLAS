#! /bin/sh
#
#  check-ilp64.sh LIBDIR BUILDDIR [LIBNAME [front|- [INCDIR]]]: the ILP64
#  interfaces (dgemm_64_, cblas_dgemm_64, ...) of the shared library
#  lib$LIBNAME (default ulmblas) in LIBDIR, built with ILP64=1
#  (interfaces/blas/shared, or the dispatching library
#  interfaces/blas/dispatch: "front").  LIBNAME ulmblas64: libulmblas64,
#  the same interfaces under the usual names (dgemm_, cblas_dgemm, ...;
#  INTERFACE64=1, or the front of the dispatching library); INCDIR holds
#  its cblas.h:
#
#  1. the exported names: every name of ilp64/symbols.txt (the ILP64 names
#     of the reference BLAS and CBLAS), for the dispatching library
#     without the error handlers xerbla_64_ and cblas_xerbla_64, which it
#     does not forward;
#  2. the Netlib test programs of test/F77, compiled with 64-bit INTEGER
#     (-fdefault-integer-8) and every BLAS name renamed by the preprocessor
#     (DGEMM -> DGEMM_64, i.e. dgemm_64_; also LSAME and their XERBLA;
#     lines up to 132 columns, as the names grow);
#  3. ilp64/ilp64.c: the ILP64 against the LP64 interfaces, 64-bit
#     arguments, the error handlers of the program; for libulmblas64
#     ilp64/plain64.c (it has no LP64 interfaces) and the names are not
#     renamed in 1. and 2.
#
#  BUILDDIR receives the programs and their output.  Needs FC and CC
#  (environment, default gfortran and cc), FFLAGS and CFLAGS are passed
#  on; NM (default nm).  Used by "make check-shared".
#
LIBDIR=`cd "$1" && pwd`
mkdir -p "$2" || exit 1
B=`cd "$2" && pwd`
LIBNAME=${3:-ulmblas}
FRONT=${4:-}
INCDIR=
if [ -n "${5:-}" ]; then
    INCDIR=`cd "$5" && pwd` || exit 1
fi
PLAIN=no
case $LIBNAME in *64) PLAIN=yes ;; esac
FC=${FC:-gfortran}
CC=${CC:-cc}
NM=${NM:-nm}
cd "`dirname "$0"`" || exit 1
T=`pwd`

case `uname -s` in
    MINGW*|MSYS*)
        LINK="-L$LIBDIR -l$LIBNAME -Wl,--export-all-symbols"
        PATH="$LIBDIR:$PATH"; export PATH
        LIB=$LIBDIR/lib$LIBNAME.dll ;;
    Darwin)
        LINK="-L$LIBDIR -l$LIBNAME -Wl,-rpath,$LIBDIR"
        LIB=$LIBDIR/lib$LIBNAME.dylib ;;
    *)  LINK="-L$LIBDIR -l$LIBNAME -Wl,-rpath,$LIBDIR"
        LIB=$LIBDIR/lib$LIBNAME.so ;;
esac

n_pass=0
n_fail=0

pass()
{
    echo "PASS  $1"
    n_pass=`expr $n_pass + 1`
}

fail()
{
    echo "FAIL  $1"
    n_fail=`expr $n_fail + 1`
}

#  1. exported names (Windows: the export table of objdump -p)
if [ $PLAIN = yes ]; then
    grep -v '^#' ilp64/symbols.txt \
        | sed -e 's/_64_$/_/' -e 's/^\(cblas_.*\)_64$/\1/' | sort
else
    grep -v '^#' ilp64/symbols.txt | sort
fi > "$B/expected.txt"
if [ "$FRONT" = front ]; then
    grep -v -E '^(xerbla_64_|cblas_xerbla_64|xerbla_|cblas_xerbla)$' \
        "$B/expected.txt" > "$B/expected.tmp" \
        && mv "$B/expected.tmp" "$B/expected.txt"
fi
case `uname -s` in
    MINGW*|MSYS*)
        objdump -p "$LIB" | awk '/\[Ordinal\/Name Pointer\] Table/ {t = 1; next}
                                 t && NF == 0 {t = 0}
                                 t {print $NF}' ;;
    Darwin)
        $NM -gU "$LIB" | awk '{print $3}' | sed 's/^_//' ;;
    *)  $NM -D --defined-only "$LIB" | awk '{print $3}' ;;
esac | sort -u > "$B/exported.txt"
missing=`comm -23 "$B/expected.txt" "$B/exported.txt"`
if [ -n "$missing" ]; then
    fail "exports: `echo $missing | wc -w` names missing: `echo $missing | cut -c1-200`"
else
    pass "exports (`wc -l < "$B/expected.txt" | tr -d ' '` names)"
fi

#  2. Netlib test programs with 64-bit INTEGER, the names renamed
FFLAGS_NETLIB="-w -fdefault-integer-8 -cpp -ffixed-line-length-132"
if echo | $FC -fallow-argument-mismatch -x f77 -c - -o /dev/null \
        >/dev/null 2>&1; then
    FFLAGS_NETLIB="$FFLAGS_NETLIB -fallow-argument-mismatch"
fi
RENAME=`grep -v '^#' ilp64/symbols.txt | sed -n 's/_64_$//p' \
        | tr 'a-z' 'A-Z' | sed 's/.*/-D&=&_64/'`
[ $PLAIN = yes ] && RENAME=

#  run PROGRAM LEVEL INPUT: run in $B, judged as by check-shared.sh
run()
{
    program=$1
    level=$2
    input=$3
    summary=`echo "$program" | tr 'a-z' 'A-Z'`.SUMM
    rm -f "$B/$summary"
    (cd "$B" && "./$program" < "$input" > "$program.out" 2>&1)
    status=$?
    if [ -f "$B/$summary" ]; then
        cat "$B/$summary" >> "$B/$program.out"
    fi
    out=$B/$program.out
    message=
    if [ $status -ne 0 ]; then
        message="exit status $status"
    elif grep -q -E 'FAIL|FATAL|SUSPECT|ILLEGAL|NOT TESTED|ERROR [^E]' "$out"
    then
        message=`grep -E 'FAIL|FATAL|SUSPECT|ILLEGAL|NOT TESTED|ERROR [^E]' \
                 "$out" | head -n 1 | sed 's/^ *//'`
    elif [ "$level" = 1 ]; then
        tested=`grep -c 'Test of subprogram number' "$out"`
        passed=`grep -c -- '----- PASS -----' "$out"`
        if [ "$tested" -eq 0 ] || [ "$tested" -ne "$passed" ]; then
            message="$passed of $tested subprograms passed"
        fi
    elif ! grep -q 'END OF TESTS' "$out"; then
        message="incomplete output (no END OF TESTS)"
    elif ! grep -q 'PASSED THE' "$out"; then
        message="no routine passed"
    fi
    if [ -n "$message" ]; then
        fail "$program (ILP64): $message (see $out)"
    else
        pass "$program (ILP64)"
    fi
}

for p in sblat1 dblat1 cblat1 zblat1 sblat2 dblat2 cblat2 zblat2 \
         sblat3 dblat3 cblat3 zblat3; do
    level=`echo $p | sed 's/.*blat//'`
    if ! $FC $FFLAGS $FFLAGS_NETLIB $RENAME -o "$B/$p" F77/$p.f $LINK; then
        fail "$p (ILP64): not built"
        continue
    fi
    if [ "$level" = 1 ]; then
        input=/dev/null
    else
        cp F77/$p.dat.in "$B/$p.dat"
        input=$p.dat
    fi
    run $p $level $input
done

#  3. ILP64 against LP64; libulmblas64 against simple loops
if [ $PLAIN = yes ]; then
    if ! $CC $CFLAGS -I"$INCDIR" -o "$B/plain64" ilp64/plain64.c $LINK -lm
    then
        fail "plain64: not built"
    elif (cd "$B" && ./plain64 > plain64.out 2>&1); then
        pass "plain64"
    else
        fail "plain64: `grep FAIL "$B/plain64.out" | head -n 3 | tr '\n' ' '`(see $B/plain64.out)"
    fi
elif ! $CC $CFLAGS -o "$B/ilp64" ilp64/ilp64.c $LINK -lm; then
    fail "ilp64: not built"
elif (cd "$B" && ./ilp64 > ilp64.out 2>&1); then
    pass "ilp64"
else
    fail "ilp64: `grep FAIL "$B/ilp64.out" | head -n 3 | tr '\n' ' '`(see $B/ilp64.out)"
fi

echo "check-ilp64: PASS $n_pass, FAIL $n_fail"
[ $n_fail -eq 0 ]
