#!/bin/sh
#
#  make tune: measures the GEMM candidates of this build (ARCHFLAGS) on
#  this machine and generates the selection table for its platform (see
#  tune/README.md).
#
#  Environment (from make): CXX, ARCHFLAGS, OPTFLAGS; optional:
#
#      PASSES     passes through the grid (default 3), each in separate
#                 program runs
#      ROUNDS     samples per shape and candidate in a pass (default 5)
#      CHUNK      shapes per program run (default 50)
#      LOADMAX    a program run starts only if the 1-minute load average is
#                 below LOADMAX (default: 1.5)
#      PAUSE      seconds of pause before a program run (default 5)
#      PLATFORM   platform name of the table (default: detected, see
#                 ulmblas/config/platform.h)
#      GRID       values of m, n and k of the grid, comma separated
#                 (default: 32,64,128,256,512,1100,2300)
#      TUNESET    the candidate set (e.g. cfloat-x86-avx512) if the
#                 conditions of several hold (default: the first)
#
set -e

TOP=$(cd "$(dirname "$0")/.." && pwd)
BUILD=$TOP/tune/build
PASSES=${PASSES:-3}
ROUNDS=${ROUNDS:-5}
CHUNK=${CHUNK:-50}
LOADMAX=${LOADMAX:-1.5}
PAUSE=${PAUSE:-5}

mkdir -p "$BUILD"

#
#  Build: the candidates of the build, the measurement program and (macOS)
#  the probe of the thermal state.
#
if [ -n "${GRID:-}" ]; then
    export ULMBLAS_TUNE_GRID=$GRID
fi
python3 "$TOP/tune/gemm/gentable.py" candidates "$BUILD/candidates.h"
SETFLAGS=
if [ -n "${TUNESET:-}" ]; then
    SETFLAGS="-DULMBLAS_TUNE_SET_INDEX=$(python3 "$TOP/tune/gemm/gentable.py" \
        index "$TUNESET")"
fi
#  the assembly kernels with AVX2 (as make.inc builds them)
ASM=
if echo | $CXX $ARCHFLAGS -dM -E -x c++ - 2>/dev/null | grep -q '__AVX2__ 1' &&
   echo | $CXX $ARCHFLAGS -dM -E -x c++ - 2>/dev/null | grep -q '__x86_64__ 1'
then
    ASM=$TOP/ulmblas/level3/ukernel/openblas/asm/haswell_4x8.S
fi
echo "tune: building the measurement program (ARCHFLAGS = $ARCHFLAGS)"
$CXX -std=c++11 $OPTFLAGS $ARCHFLAGS $SETFLAGS -I"$TOP" -I"$BUILD" \
    "$TOP/tune/gemm/measure.cc" $ASM -o "$BUILD/measure"
THERMAL=
if [ "$(uname)" = Darwin ]; then
    cc -O2 -framework Foundation "$TOP/tune/thermal.m" -o "$BUILD/thermal"
    THERMAL=$BUILD/thermal
fi

SET=$("$BUILD/measure" set) || {
    echo "tune: no GEMM candidates for this build (ARCHFLAGS = $ARCHFLAGS);"
    echo "      see tune/gemm/candidates.py"
    exit 1
}
if [ -z "${PLATFORM:-}" ]; then
    PLATFORM=$("$BUILD/measure" platform)
fi
SHAPES=$("$BUILD/measure" list | wc -l | tr -d ' ')
RUNS=$(( (SHAPES + CHUNK - 1) / CHUNK * PASSES ))
MINUTES=$(( (RUNS * (CHUNK * 4 / 10 + PAUSE) + 59) / 60 ))

cat <<EOF

  make tune measures GEMM on this machine: candidate set $SET,
  platform $PLATFORM, $SHAPES shapes, $PASSES passes, at least $MINUTES minutes.

  Keep the machine idle until it is done: close other programs, no builds,
  no video calls; connect a laptop to the power supply.  The measurement
  waits while the load average is above $LOADMAX or the machine is hot.

EOF

thermal_state() {
    if [ -n "$THERMAL" ]; then "$THERMAL" | awk '{print $3}'; else echo nominal; fi
}
load1() {
    if [ "$(uname)" = Darwin ]; then
        sysctl -n vm.loadavg | awk '{print $2}'
    else
        awk '{print $1}' /proc/loadavg
    fi
}
#  wait until the machine is idle and cool
idle() {
    said=
    until [ "$(thermal_state)" = nominal ] &&
          awk -v l="$(load1)" -v x="$LOADMAX" 'BEGIN { exit !(l < x) }'; do
        if [ -z "$said" ]; then
            echo "tune: waiting (load $(load1), thermal state $(thermal_state))"
            said=yes
        fi
        sleep 5
    done
    sleep "$PAUSE"
}

DATA=$BUILD/$PLATFORM-$SET.tsv
: > "$DATA"
run=0
pass=1
while [ $pass -le "$PASSES" ]; do
    first=0
    while [ $first -lt "$SHAPES" ]; do
        run=$((run + 1))
        try=1
        while :; do
            idle
            t0=$(thermal_state)
            "$BUILD/measure" run $first "$CHUNK" "$ROUNDS" > "$BUILD/chunk.tsv"
            t1=$(thermal_state)
            l1=$(load1)
            if [ "$t0" = nominal ] && [ "$t1" = nominal ] &&
               awk -v l="$l1" -v x="$LOADMAX" 'BEGIN { exit !(l < 2 * x) }'; then
                break
            fi
            echo "tune: run $run repeated (thermal state $t0 -> $t1, load $l1)"
            try=$((try + 1))
            if [ $try -gt 5 ]; then
                echo "tune: the machine is not idle; giving up"
                exit 1
            fi
        done
        cat "$BUILD/chunk.tsv" >> "$DATA"
        echo "tune: run $run of $RUNS done (pass $pass, shapes $first to $((first + CHUNK - 1)))"
        first=$((first + CHUNK))
    done
    pass=$((pass + 1))
done

RESULTS=$TOP/tune/results/$PLATFORM
mkdir -p "$RESULTS"
cp "$DATA" "$RESULTS/$SET.tsv"
python3 "$TOP/tune/gemm/gentable.py" table "$SET" "$PLATFORM" "$RESULTS/$SET.tsv" \
    "$TOP/ulmblas/level3/gemm/tables" "$RESULTS/$SET.md"
if command -v clang-format > /dev/null 2>&1; then
    clang-format -i "$TOP"/ulmblas/level3/gemm/tables/*.h
fi
echo
echo "tune: done.  Table: ulmblas/level3/gemm/tables/, report: tune/results/$PLATFORM/$SET.md"
echo "      Rebuild ulmBLAS to use the table."
