#!/bin/sh
#
#  make tune-params: measures the parameters of ulmblas/config/tuned.h
#  (threads, variants and block sizes; group by group) on this machine and
#  generates ulmblas/config/tuned/PLATFORM.h (see tune/README.md).
#
#  Environment (from make): CXX, ARCHFLAGS, OPTFLAGS, SIMDFLAGS, OMPFLAGS,
#  OMPLIBS (OPENMP=1: also the group potrf); optional:
#
#      TUNEGROUPS  the groups, in this order (default: all, see
#                  tune/params/measure.cc)
#      TYPES       the types (default: s d c z)
#      PASSES      passes through the problems (default 2)
#      ROUNDS      samples per problem and setting in a pass (default 5)
#      CHUNK       problems per program run (default 6)
#      LOADMAX     a group starts only if the 1-minute load average is below
#                  LOADMAX (default 1.5; at most 10 minutes of waiting)
#      PAUSE       seconds of pause before a program run (default 2)
#      PLATFORM    platform name (default: detected)
#      VERIFY      0: no comparison of defaults and tuned values at the end;
#                  only: the comparison only (the parameters of the build)
#
set -e

TOP=$(cd "$(dirname "$0")/.." && pwd)
BUILD=$TOP/tune/build/params
TUNEGROUPS=${TUNEGROUPS:-"gemm-threads gemm-partition trsm-kb trmm-kb trsm-variant trmm-variant level2 potrf"}
TYPES=${TYPES:-"s d c z"}
PASSES=${PASSES:-2}
ROUNDS=${ROUNDS:-5}
CHUNK=${CHUNK:-6}
LOADMAX=${LOADMAX:-1.5}
PAUSE=${PAUSE:-2}
VERIFY=${VERIFY:-1}
PY="python3 $TOP/tune/params/params.py"
TUNEDDIR=$TOP/ulmblas/config/tuned

#  the environment variables that override parameters would falsify the
#  measurements; the threads of the pool are those of the environment
#  (ULMBLAS_NUM_THREADS, OMP_NUM_THREADS, otherwise all CPUs)
unset ULMBLAS_TRSM_INV ULMBLAS_TRSM_SHARED ULMBLAS_TRSM_MINP \
      ULMBLAS_TRMM_SHARED ULMBLAS_TRMM_MINP ULMBLAS_AFFINITY

mkdir -p "$BUILD"

ASM=
if echo | $CXX $ARCHFLAGS -dM -E -x c++ - 2>/dev/null | grep -q '__AVX2__ 1' &&
   echo | $CXX $ARCHFLAGS -dM -E -x c++ - 2>/dev/null | grep -q '__x86_64__ 1'
then
    ASM=$TOP/ulmblas/level3/ukernel/openblas/asm/haswell_4x8.S
fi
build() {
    $CXX -std=c++11 $OPTFLAGS $ARCHFLAGS $SIMDFLAGS $OMPFLAGS -pthread \
        -I"$TOP" "$TOP/tune/params/measure.cc" $ASM -o "$BUILD/measure" \
        $OMPLIBS -pthread
}
echo "tune-params: building the measurement program (ARCHFLAGS = $ARCHFLAGS)"
build
THERMAL=
if [ "$(uname)" = Darwin ]; then
    cc -O2 -framework Foundation "$TOP/tune/thermal.m" -o "$BUILD/thermal"
    THERMAL=$BUILD/thermal
fi

INFO=$("$BUILD/measure" info)
CONDITION=$(echo "$INFO" | cut -f2)
POOL=$(echo "$INFO" | cut -f3)
if [ -z "${PLATFORM:-}" ]; then
    PLATFORM=$(echo "$INFO" | cut -f1)
fi
RESULTS=$TOP/tune/results/$PLATFORM
mkdir -p "$RESULTS"
$PY pool "$RESULTS" "$POOL"

cat <<EOF

  make tune-params measures the parameters of ulmblas/config/tuned.h on
  this machine: platform $PLATFORM, $POOL threads, groups $TUNEGROUPS.

  Keep the machine idle until it is done.

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
#  before a group: wait until the machine is idle (at most 10 minutes: the
#  load average of the measurement itself decays slowly)
idle_group() {
    waited=0
    until awk -v l="$(load1)" -v x="$LOADMAX" 'BEGIN { exit !(l < x) }'; do
        if [ $waited -ge 600 ]; then
            echo "tune-params: load $(load1) after 10 minutes, going on"
            return
        fi
        sleep 10
        waited=$((waited + 10))
    done
}
#  before a run: cool (macOS) and a pause
idle_run() {
    until [ "$(thermal_state)" = nominal ]; do
        sleep 5
    done
    sleep "$PAUSE"
}

#  runs MODE (run or verify) of a group for all types and passes into FILE
measure_group() {
    mode=$1 group=$2 file=$3
    pass=1
    while [ $pass -le "$PASSES" ]; do
        for t in $TYPES; do
            count=$("$BUILD/measure" count "$group" "$t")
            first=0
            while [ "$first" -lt "$count" ]; do
                idle_run
                "$BUILD/measure" "$mode" "$group" "$t" "$ROUNDS" "$first" \
                    "$CHUNK" >> "$file"
                first=$((first + CHUNK))
            done
        done
        echo "tune-params: $group ($mode), pass $pass of $PASSES done"
        pass=$((pass + 1))
    done
}

for group in $TUNEGROUPS; do
    [ "$VERIFY" = only ] && break
    #  no problems: nothing to measure on this machine (e.g. the threads of
    #  level 3 with SME or with one thread)
    if [ "$("$BUILD/measure" count "$group" d)" -eq 0 ]; then
        echo "tune-params: $group: nothing to measure here"
        continue
    fi
    idle_group
    DATA=$BUILD/$group.tsv
    : > "$DATA"
    measure_group run "$group" "$DATA"
    PARAM=$BUILD/param.txt
    : > "$PARAM"
    for t in s d c z; do
        echo $t >> "$PARAM"
        "$BUILD/measure" param $t >> "$PARAM"
    done
    cp "$DATA" "$RESULTS/params-$group.tsv"
    $PY fit "$group" "$PLATFORM" "$DATA" "$PARAM" "$RESULTS"
    $PY generate "$PLATFORM" "$CONDITION" "$RESULTS" "$TUNEDDIR"
    if command -v clang-format > /dev/null 2>&1; then
        clang-format -i "$TUNEDDIR"/*.h
    fi
    #  the next groups are measured with the values of this one
    build
done

if [ "$VERIFY" != 0 ]; then
    DATA=$BUILD/verify.tsv
    : > "$DATA"
    for group in $TUNEGROUPS; do
        if [ "$("$BUILD/measure" count "$group" d)" -eq 0 ]; then
            continue
        fi
        idle_group
        measure_group verify "$group" "$DATA"
    done
    cp "$DATA" "$RESULTS/params-verify.tsv"
    $PY verify "$PLATFORM" "$DATA" "$RESULTS"
fi

echo
echo "tune-params: done.  Parameters: ulmblas/config/tuned/$PLATFORM.h,"
echo "             report: tune/results/$PLATFORM/params.md."
echo "             Rebuild ulmBLAS to use them."
