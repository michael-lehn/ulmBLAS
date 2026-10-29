#!/bin/sh
#
#  run.sh: single-threaded GEMM benchmark of ulmBLAS and of the other BLAS
#  libraries found on this machine.  Usually started with 'make bench-gemm'
#  (see README.md in this directory), which builds libcblas.a of ulmBLAS
#  with ARCHFLAGS first.  Settings (environment or make variables):
#
#      PREC      data type: s, d, c, z (SGEMM, DGEMM, CGEMM, ZGEMM; default d)
#      LIBS      'auto' (default): ulmBLAS and every library found;
#                or a list of names, e.g. "ulmblas openblas": each must be
#                found, otherwise the run stops
#      SIZES     m = n = k (default 50 100 ... 4000, 80 sizes)
#      SAMPLES   timed samples per size, median reported (default 3)
#      ROUNDS    rounds over all libraries, order rotated (default 1)
#      CPU       Linux: run every program on this CPU (taskset); default none
#      OUT       result directory (default bench/gemm/results/DATE-PREC)
#      OPENBLAS_DIR, BLIS_DIR, AOCL_DIR (or AOCL_ROOT), MKLROOT, EIGEN_DIR
#                installation directories; they take precedence over
#                pkg-config and the default search
#      CC, CXX, ARCHFLAGS   compilers; ARCHFLAGS of the ulmBLAS build (from
#                make), recorded only
#
#  Libraries: ulmblas (this tree), openblas, blis, aocl (AOCL-BLAS), mkl
#  (oneMKL, sequential), accelerate (macOS), eigen (C++, not CBLAS).  A
#  library counts as found only if the benchmark program builds, links and
#  runs with it and reports the expected library (benchx -i); it is never
#  replaced by another one.  Every library is limited to one thread.
#
set -u
TOP=$(cd "$(dirname "$0")/../.." && pwd)
D=$TOP/bench/gemm
PREC=${PREC:-d}
LIBS=${LIBS:-auto}
SIZES=${SIZES:-$(awk 'BEGIN { for (n=50; n<=4000; n+=50) print n }')}
SAMPLES=${SAMPLES:-3}
ROUNDS=${ROUNDS:-1}
CPU=${CPU:-}
CC=${CC:-cc}
CXX=${CXX:-c++}
case $PREC in
    s|d|c|z) ;;
    *) echo "error: PREC must be s, d, c or z (got '$PREC')" >&2; exit 2;;
esac
NAME=$(echo $PREC | tr sdcz SDCZ)GEMM
OUT=${OUT:-$D/results/$(date +%Y%m%d-%H%M%S)-${PREC}gemm}
mkdir -p "$OUT/bin" "$OUT/raw" || exit 1
DET=$OUT/detection.txt
: > "$DET"
note() { echo "$*" | tee -a "$DET"; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE VECLIB_MAXIMUM_THREADS=1
OS=$(uname -s)
ARCH=$(uname -m)

{
    echo "# $NAME benchmark, $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# host: $(uname -n), $OS $(uname -r) $ARCH"
    echo "# CC: $($CC --version 2>&1 | head -1)"
    echo "# CXX: $($CXX --version 2>&1 | head -1)"
    echo "# ulmBLAS ARCHFLAGS: ${ARCHFLAGS:-}"
    echo "# threads: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"
    echo "# sizes: $(echo $SIZES | tr '\n' ' ')"
    echo "# samples $SAMPLES, rounds $ROUNDS, cpu ${CPU:-not pinned}"
} >> "$DET"

PIN=
if [ -n "$CPU" ]; then
    if command -v taskset >/dev/null 2>&1; then
        PIN="taskset -c $CPU"
        if ! $PIN true 2>/dev/null; then
            echo "error: CPU=$CPU cannot be used; allowed: $(taskset -pc $$ 2>/dev/null | sed 's/.*: //')" >&2
            exit 2
        fi
    else
        note "note: CPU=$CPU ignored (no taskset on this system)"
    fi
fi

#  candidate(NAME): sets HOW (where the library was found), KIND (c: benchx,
#  cxx: Eigen), CFLAGS_L, LIBS_L; returns 1 if there is no candidate
rpath() { [ -n "$1" ] && echo "-Wl,-rpath,$1"; }
pkg() {
    command -v pkg-config >/dev/null 2>&1 && pkg-config --exists "$1" 2>/dev/null
}
candidate() {
    HOW=; KIND=c; CFLAGS_L=; LIBS_L=
    case $1 in
    ulmblas)
        [ -f "$TOP/libcblas.a" ] || return 1
        HOW="this tree ($TOP/libcblas.a)"
        CFLAGS_L="-I$TOP"; LIBS_L="$TOP/libcblas.a";;
    openblas)
        if [ -n "${OPENBLAS_DIR:-}" ]; then
            HOW="OPENBLAS_DIR=$OPENBLAS_DIR"
            CFLAGS_L="-I$OPENBLAS_DIR/include"
            LIBS_L="-L$OPENBLAS_DIR/lib $(rpath $OPENBLAS_DIR/lib) -lopenblas -lpthread -lm"
        elif pkg openblas; then
            HOW="pkg-config openblas $(pkg-config --modversion openblas)"
            CFLAGS_L=$(pkg-config --cflags openblas)
            LIBS_L="$(pkg-config --libs openblas) $(rpath "$(pkg-config --variable=libdir openblas)")"
        elif [ "$OS" = Darwin ] && command -v brew >/dev/null 2>&1 \
             && P=$(brew --prefix openblas 2>/dev/null) && [ -d "$P/lib" ]; then
            HOW="Homebrew ($P)"
            CFLAGS_L="-I$P/include"; LIBS_L="-L$P/lib $(rpath $P/lib) -lopenblas"
        else
            HOW="default compiler and linker paths"
            LIBS_L="-lopenblas -lpthread -lm"
        fi;;
    blis)
        if [ -n "${BLIS_DIR:-}" ]; then
            HOW="BLIS_DIR=$BLIS_DIR"
            CFLAGS_L="-I$BLIS_DIR/include/blis -I$BLIS_DIR/include"
            LIBS_L="-L$BLIS_DIR/lib $(rpath $BLIS_DIR/lib) -lblis -lpthread -lm"
        elif pkg blis; then
            HOW="pkg-config blis $(pkg-config --modversion blis)"
            CFLAGS_L=$(pkg-config --cflags blis)
            LIBS_L="$(pkg-config --libs blis) $(rpath "$(pkg-config --variable=libdir blis)") -lpthread -lm"
        else
            HOW="default compiler and linker paths"
            CFLAGS_L="-I/usr/include/blis -I/usr/local/include/blis"
            LIBS_L="-lblis -lpthread -lm"
        fi;;
    aocl)
        A=${AOCL_DIR:-${AOCL_ROOT:-}}
        [ -n "$A" ] || return 1
        HOW="AOCL_DIR/AOCL_ROOT=$A"
        CFLAGS_L="-I$A/include -I$A/include/blis"
        LIBS_L="-L$A/lib $(rpath $A/lib) -lblis -lpthread -lm";;
    mkl)
        [ -n "${MKLROOT:-}" ] || return 1
        L=$MKLROOT/lib; [ -d "$MKLROOT/lib/intel64" ] && L=$MKLROOT/lib/intel64
        HOW="MKLROOT=$MKLROOT (sequential, LP64)"
        CFLAGS_L="-I$MKLROOT/include '-DBENCH_CBLAS_HEADER=<mkl_cblas.h>'"
        if [ "$OS" = Darwin ]; then
            LIBS_L="-L$L $(rpath $L) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm"
        else
            LIBS_L="-L$L $(rpath $L) -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
        fi;;
    accelerate)
        [ "$OS" = Darwin ] || return 1
        HOW="macOS framework"
        CFLAGS_L="'-DBENCH_CBLAS_HEADER=<Accelerate/Accelerate.h>' -Wno-deprecated-declarations"
        LIBS_L="-framework Accelerate";;
    eigen)
        KIND=cxx
        if [ -n "${EIGEN_DIR:-}" ]; then
            HOW="EIGEN_DIR=$EIGEN_DIR"; CFLAGS_L="-I$EIGEN_DIR"
        elif pkg eigen3; then
            HOW="pkg-config eigen3 $(pkg-config --modversion eigen3)"
            CFLAGS_L=$(pkg-config --cflags eigen3)
        else
            return 1
        fi
        case $ARCH in
            x86_64|amd64) CFLAGS_L="$CFLAGS_L ${EIGEN_FLAGS:--march=native}";;
            *)            CFLAGS_L="$CFLAGS_L ${EIGEN_FLAGS:--mcpu=native}";;
        esac;;
    *)
        return 1;;
    esac
}

#  the identity the probe must report (benchx -i), see benchx.c
expected() {
    case $1 in
        ulmblas)    echo '^# library ulmBLAS';;
        openblas)   echo '^# library OpenBLAS';;
        blis)       echo '^# library BLIS ';;
        aocl)       echo '^# library .*AOCL';;
        mkl)        echo '^# library .*(oneMKL|Math Kernel Library)';;
        accelerate) echo 'from .*/Accelerate\.framework/';;
        eigen)      echo '^# library Eigen';;
    esac
}

#  build(NAME): builds $OUT/bin/bench-NAME, checks its identity
build() {
    b=$OUT/bin/bench-$1
    if [ "$KIND" = cxx ]; then
        cmd="$CXX -std=c++14 -O3 -DPREC=$PREC $CFLAGS_L $D/bench-eigen.cpp -o $b"
    elif [ "$1" = ulmblas ]; then
        #  libcblas.a needs the C++ runtime: linked by the C++ compiler
        cmd="$CC -O2 -DPREC=$PREC -DBENCH_LIB_$1 $CFLAGS_L -c $D/benchx.c -o $b.o && $CXX $b.o $LIBS_L -lm -o $b"
    else
        dl=; [ "$OS" = Linux ] && dl=-ldl
        cmd="$CC -O2 -DPREC=$PREC -DBENCH_LIB_$1 $CFLAGS_L $D/benchx.c $LIBS_L $dl -lm -o $b"
    fi
    echo "$1: $cmd" >> "$OUT/build.log"
    if ! sh -c "$cmd" >> "$OUT/build.log" 2>&1; then
        WHY="does not compile or link (see build.log)"; return 1
    fi
    if ! ID=$("$b" -i 2>&1); then
        WHY="the program does not run: $(echo "$ID" | tail -1)"; return 1
    fi
    if ! echo "$ID" | grep -qE "$(expected $1)"; then
        WHY="unexpected library: $(echo "$ID" | grep -E '^# library|from' | tr '\n' ' ')"; return 1
    fi
    if [ "$1" = blis ] && echo "$ID" | grep -q AOCL; then
        WHY="this is AOCL-BLAS, not BLIS (use LIBS=aocl and AOCL_DIR)"; return 1
    fi
    return 0
}

KNOWN="ulmblas openblas blis aocl mkl accelerate eigen"
if [ "$LIBS" = auto ]; then WANT=$KNOWN; else WANT=$LIBS; fi
SELECTED=
note "libraries ($LIBS):"
for l in $WANT; do
    case " $KNOWN " in
        *" $l "*) ;;
        *) note "  $l: unknown (known: $KNOWN)"; exit 2;;
    esac
    if ! candidate $l; then
        case $l in
            ulmblas)    WHY="libcblas.a not built (run 'make bench-gemm')";;
            aocl)       WHY="AOCL_DIR or AOCL_ROOT not set";;
            mkl)        WHY="MKLROOT not set";;
            accelerate) WHY="only on macOS";;
            eigen)      WHY="neither EIGEN_DIR nor pkg-config eigen3";;
        esac
    elif build $l; then
        note "  $l: selected, found via $HOW"
        echo "$ID" | sed 's/^#/     /' | tee -a "$DET"
        SELECTED="$SELECTED $l"
        continue
    fi
    note "  $l: skipped, $WHY${HOW:+ (tried: $HOW)}"
    if [ "$LIBS" != auto ]; then
        echo "error: library '$l' was requested but is not usable" >&2
        exit 1
    fi
done
[ -n "$SELECTED" ] || { echo "error: no library to benchmark" >&2; exit 1; }
case " $SELECTED " in
*" ulmblas "*)
    #  the kernels ulmBLAS selects with its build flags (compile-time
    #  selection: the same headers and ARCHFLAGS as libcblas.a)
    C=$TOP/interfaces/blas/C/build.cblas/compile.cmd
    [ -f "$C" ] && note "ulmBLAS compile command: $(cat "$C")"
    if $CXX -std=c++11 ${ARCHFLAGS:-} -I"$TOP" "$D/ulmblas-kernels.cc" \
            -o "$OUT/bin/ulmblas-kernels" >> "$OUT/build.log" 2>&1; then
        "$OUT/bin/ulmblas-kernels" | sed 's/^/  /' | tee -a "$DET"
    else
        note "  (ulmblas-kernels did not compile, see build.log)"
    fi;;
esac


rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
note "running $NAME for:$SELECTED"
for r in $(awk -v n=$ROUNDS 'BEGIN { for (i=1; i<=n; ++i) print i }'); do
    for l in $(rot $r $SELECTED); do
        echo "  round $r: $l"
        $PIN "$OUT/bin/bench-$l" $SAMPLES $SIZES > "$OUT/raw/$l-$r.txt" \
            || { echo "error: bench-$l failed (round $r)" >&2; exit 1; }
    done
done

if grep -q WRONG "$OUT"/raw/*.txt; then
    note "WRONG results: $(grep -l WRONG "$OUT"/raw/*.txt | tr '\n' ' ')"
fi
if command -v python3 >/dev/null 2>&1; then
    python3 "$D/summarize.py" "$OUT" > "$OUT/summary.md" && cat "$OUT/summary.md"
else
    echo "note: python3 not found, no summary.md"
fi
GNUPLOT=${GNUPLOT:-gnuplot}
if command -v $GNUPLOT >/dev/null 2>&1 && [ -f "$OUT/gflops.dat" ]; then
    {
        echo "set terminal svg size 900,560; set output '$OUT/gflops.svg'"
        echo "set title '$NAME, single-threaded, m = n = k'; set xlabel 'n'; set ylabel 'GFLOP/s'"
        echo "set key outside right; set grid; set yrange [0:*]"
        i=2; printf "plot "
        for l in $SELECTED; do
            printf "'%s' using 1:%d with linespoints title '%s', " "$OUT/gflops.dat" $i $l
            i=$((i+1))
        done | sed 's/, $//'
        echo
    } > "$OUT/gflops.gp"
    $GNUPLOT "$OUT/gflops.gp" && echo "plot: $OUT/gflops.svg"
fi
echo "results: $OUT (detection.txt, raw/, summary.md)"
