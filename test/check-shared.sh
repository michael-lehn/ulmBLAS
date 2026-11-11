#! /bin/sh
#
#  check-shared.sh LIBDIR INCDIR BUILDDIR [LIBNAME [f77]]: the Netlib BLAS
#  test programs of test/F77 and test/C, linked against the shared library
#  lib$LIBNAME (default ulmblas, interfaces/blas/shared) in LIBDIR instead
#  of the static libraries.  With "f77" the CBLAS programs of levels 2 and
#  3 are skipped: their error handler reads RowMajorStrg, which the
#  dispatching library (interfaces/blas/dispatch) does not export.
#  INCDIR holds the generated cblas.h, BUILDDIR receives the programs and
#  their output.  The outputs are judged as by run-tests.sh.  Used by
#  "make check-shared"; needs FC and CC (environment, default gfortran and
#  cc), FFLAGS and CFLAGS are passed on.
#
#  The test programs replace XERBLA and cblas_xerbla to check the error
#  exits; the library must call these replacements, not its own handlers.
#
LIBDIR=`cd "$1" && pwd`
INCDIR=`cd "$2" && pwd`
mkdir -p "$3" || exit 1
B=`cd "$3" && pwd`
FC=${FC:-gfortran}
CC=${CC:-cc}
cd "`dirname "$0"`" || exit 1
T=`pwd`

FFLAGS_NETLIB="-w"
if echo | $FC -fallow-argument-mismatch -x f77 -c - -o /dev/null \
        >/dev/null 2>&1; then
    FFLAGS_NETLIB="$FFLAGS_NETLIB -fallow-argument-mismatch"
fi
LIBNAME=${4:-ulmblas}
ONLY_F77=${5:-}
case `uname -s` in
    MINGW*|MSYS*)
        #  Windows: the import library at link time, the DLLs by PATH; the
        #  program exports its XERBLA so that the library finds it
        LINK="-L$LIBDIR -l$LIBNAME -Wl,--export-all-symbols"
        PATH="$LIBDIR:$PATH"; export PATH ;;
    *)  LINK="-L$LIBDIR -l$LIBNAME -Wl,-rpath,$LIBDIR" ;;
esac

n_pass=0
n_fail=0

check()
{
    out=$1
    level=$2
    if grep -q -E 'FAIL|FATAL|SUSPECT|ILLEGAL|NOT TESTED|ERROR [^E]' "$out"
    then
        grep -E 'FAIL|FATAL|SUSPECT|ILLEGAL|NOT TESTED|ERROR [^E]' "$out" \
            | head -n 1 | sed 's/^ *//'
        return
    fi
    if [ "$level" = 1 ]; then
        tested=`grep -c 'Test of subprogram number' "$out"`
        passed=`grep -c -- '----- PASS -----' "$out"`
        if [ "$tested" -eq 0 ] || [ "$tested" -ne "$passed" ]; then
            echo "$passed of $tested subprograms passed"
        fi
    else
        if ! grep -q 'END OF TESTS' "$out"; then
            echo "incomplete output (no END OF TESTS)"
        elif ! grep -q 'PASSED THE' "$out"; then
            echo "no routine passed"
        fi
    fi
}

#  run PROGRAM LEVEL INPUT: run in $B, judge the output
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
    if [ $status -ne 0 ]; then
        message="exit status $status"
    else
        message=`check "$B/$program.out" "$level"`
    fi
    if [ -n "$message" ]; then
        echo "FAIL  $program: $message (see $B/$program.out)"
        n_fail=`expr $n_fail + 1`
    else
        echo "PASS  $program"
        n_pass=`expr $n_pass + 1`
    fi
}

#  Fortran 77 interface
for p in sblat1 dblat1 cblat1 zblat1 sblat2 dblat2 cblat2 zblat2 \
         sblat3 dblat3 cblat3 zblat3 ublat1; do
    level=`echo $p | sed 's/.*blat//'`
    if ! $FC $FFLAGS $FFLAGS_NETLIB -o "$B/$p" F77/$p.f $LINK; then
        echo "FAIL  $p: not built"
        n_fail=`expr $n_fail + 1`
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

#  C interface: Fortran drivers, C wrappers, error-exit tests
CFLAGS_NETLIB="-w -Wno-incompatible-pointer-types -DADD_ \
               -DBLAS_HEADER=<cblas.h> -I$INCDIR -I$T/.."
for t in s d c z; do
    for level in 1 2 3; do
        p=c_${t}blat$level
        if [ "$ONLY_F77" = f77 ] && [ $level != 1 ]; then
            continue
        fi
        objects="$B/$p.o $B/c_${t}blas$level.o"
        ok=yes
        $FC $FFLAGS $FFLAGS_NETLIB -c -o "$B/$p.o" C/$p.f || ok=no
        $CC $CFLAGS $CFLAGS_NETLIB -c -o "$B/c_${t}blas$level.o" \
            C/c_${t}blas$level.c || ok=no
        if [ $level != 1 ]; then
            for c in c_$t${level}chke auxiliary c_xerbla; do
                $CC $CFLAGS $CFLAGS_NETLIB -c -o "$B/$c.o" C/$c.c || ok=no
                objects="$objects $B/$c.o"
            done
        fi
        if [ $ok = no ] || ! $FC -o "$B/$p" $objects $LINK; then
            echo "FAIL  $p: not built"
            n_fail=`expr $n_fail + 1`
            continue
        fi
        if [ $level = 1 ]; then
            input=/dev/null
        else
            input=$T/C/${t}in$level
        fi
        run $p $level $input
    done
done

#  Extensions of the C interface: sbgemm (bfloat16), shgemm (float16)
for p in sbgemm shgemm i8gemm; do
    defs=; src=extensions/gemm16.c
    [ $p = shgemm ] && defs=-DFLOAT16
    [ $p = i8gemm ] && src=extensions/i8gemm.c
    if ! $CC $CFLAGS $defs -I$INCDIR -o "$B/c_$p" $src \
            $LINK -lm; then
        echo "FAIL  c_$p: not built"
        n_fail=`expr $n_fail + 1`
    elif (cd "$B" && "./c_$p" > "c_$p.out" 2>&1); then
        echo "PASS  c_$p"
        n_pass=`expr $n_pass + 1`
    else
        echo "FAIL  c_$p: `tail -n 1 "$B/c_$p.out"` (see $B/c_$p.out)"
        n_fail=`expr $n_fail + 1`
    fi
done

echo "check-shared: PASS $n_pass, FAIL $n_fail"
[ $n_fail -eq 0 ]
