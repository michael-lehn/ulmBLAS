#!/bin/sh
#  kernels-heim-iter.sh SRC WORKSPACE NAME=ARCHFLAGS ... [-- PROGRAM=PATH ...]:
#  a tuning iteration on heim (Intel Core i5-12500): builds ulmBLAS (source
#  tree SRC) once per NAME with the given ARCHFLAGS, links bench2.c and
#  verify-large.c (taken from the directory of this script), adds existing
#  programs PATH as variant PROGRAM, and measures all variants with the
#  methodology and sizes of kernels-heim.sh (core 3, 7 samples of at least
#  0.2 s, median; ROUNDS rounds with rotated order; first result checked).
#
#      sh kernels-heim-iter.sh SRC W \
#          ulmblas-haswell_6x8-mc288="-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_6x8 -DULMBLAS_DGEMM_MC=288" \
#          -- blis-haswell=/tmp/ulmblas-avx.tGdSUq/bench/bench-blis
#
set -u
SRC=$1; W=$2; shift 2
D=$(cd $(dirname $0) && pwd)
mkdir -p $W && cd $W || exit 1
L=$W/logs; mkdir -p $L res
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
CPU=3; ROUNDS=${ROUNDS:-3}
VARIANTS=""
: > $L/compile-commands.txt
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    v=${1%%=*}; f=${1#*=}; shift
    rm -rf v-$v && mkdir v-$v
    (cd $SRC && find . -path ./test/build -prune -o -type f -print | grep -vE '^\./(benchmark_results|investigations|test/build)/' | tar cf - -T -) | tar xf - -C v-$v
    (cd v-$v && make -j10 cblas ARCHFLAGS="$f") > $L/build-$v.log 2>&1 || fail $? "build $v"
    echo "$v: $(cat v-$v/interfaces/blas/C/build.cblas/compile.cmd)" >> $L/compile-commands.txt
    gcc -O2 -Iv-$v $D/bench2.c v-$v/libcblas.a -lstdc++ -lm -o bench-$v || fail $? "link $v"
    gcc -O2 -Iv-$v $D/verify-large.c v-$v/libcblas.a -lstdc++ -lm -o verify-$v || fail $? "link verify $v"
    rm -rf v-$v
    VARIANTS="$VARIANTS $v"
done
[ "${1:-}" = "--" ] && shift
for p in "$@"; do
    v=${p%%=*}; f=${p#*=}
    cp $f bench-$v || fail $? "copy $v"
    vf=$(dirname $f)/$(basename $f | sed 's/^bench-/verify-/')
    [ -x $vf ] && cp $vf verify-$v
    VARIANTS="$VARIANTS $v"
done
sha256sum bench-* verify-* > $L/programs.sha256

{
  echo "date $(date -Iseconds)"; uname -a; cat /etc/debian_version
  gcc --version | head -1; grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo
  echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor) no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  echo "load $(cat /proc/loadavg)"; who | wc -l
  echo "variants:$VARIANTS"
} > $L/system.txt 2>&1

SQ="7 7 7 31 31 31 64 64 64 100 100 100 128 128 128 199 199 199 256 256 256 384 384 384 512 512 512 768 768 768"
LARGE="1000 1000 1000 1024 1024 1024 1500 1500 1500 2000 2000 2000 2048 2048 2048 3000 3000 3000 4000 4000 4000 4096 4096 4096"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
HWMON=
for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = coretemp ] && HWMON=$h
done
cond() {
    t=; [ -n "$HWMON" ] && t=$(($(cat $HWMON/temp1_input)/1000))
    f=$(awk -v c=$CPU '$1=="processor"{p=$3} $1=="cpu" && $2=="MHz" && p==c {print $4}' /proc/cpuinfo)
    echo "$(date +%s),$1,$2,$3,$(cut -d' ' -f1 /proc/loadavg),$t,$f,$(who | wc -l)" >> $L/conditions.csv
}
echo "time,kind,variant,round,load1,package_temp_C,cpu3_MHz,sessions" > $L/conditions.csv
echo "start $(date -Iseconds)" >> $L/run.log
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        cond main $v $r; taskset -c $CPU ./bench-$v 7 1 $SQ $LARGE $THIN > res/main-$v-$r.txt || fail $? "main $v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
for v in $VARIANTS; do
    [ -x verify-$v ] && { taskset -c $CPU ./verify-$v 1000 1500 2000 3000 4000 4096 > res/verify-$v.txt 2>&1 || fail $? "verify $v"; }
done
echo "wrong results: $(cat res/main-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
