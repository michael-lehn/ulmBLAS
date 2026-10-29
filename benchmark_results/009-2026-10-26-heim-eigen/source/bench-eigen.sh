#!/bin/sh
#  bench-eigen.sh WORKSPACE : DGEMM benchmarks on heim (Intel Core
#  i5-12500), single threaded, core 3, with the methodology of the campaigns
#  2026-10-24-heim-avx-fma (run B) and 2026-10-26-heim-avx-fma-memory:
#  7 samples of at least 0.2 s, median; rounds with rotated order of the
#  variants.  Run check-eigen.sh first (builds bin/bench-eigen etc.).
#  Variants:
#
#    avx-cur   ulmBLAS febb852, AVX/FMA   (program of 2026-10-26, unchanged)
#    blis      BLIS 2.0                   (program of 2026-10-24, unchanged)
#    openblas  OpenBLAS 0.3.34            (program of 2026-10-24, unchanged)
#    eigen     Eigen 5.0.1, C.noalias() += A*B  (bench-eigen.cpp, new)
#
#  Kinds: main and beta0 (the shapes of 2026-10-24) in 3 rounds; sweep
#  (n = 2 ... 512, as 2026-10-26) and fine (every n = 1 ... 64) in 5 rounds.
#
set -u
W=$1; cd $W || exit 1
P=/tmp/ulmblas-intel2.2yi6R9/bench    # programs of campaign 2026-10-26
B=$W/bench; L=$W/logs; mkdir -p $B/res $L; cd $B || exit 1
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
CPU=3; ROUNDS=3; SWEEP_ROUNDS=5
VARIANTS="avx-cur blis openblas eigen"

for v in avx-cur blis openblas; do
    cp $P/bench-$v $P/verify-$v .
done
cp $W/bin/bench-eigen ./bench-eigen
cp $W/bin/verify-eigen ./verify-eigen
sha256sum bench-* verify-* > $L/programs.sha256

SQ="64 64 64 128 128 128 192 192 192 256 256 256 384 384 384 512 512 512 768 768 768 1024 1024 1024 1536 1536 1536 2048 2048 2048 3072 3072 3072 4096 4096 4096"
SMALL="7 7 7 31 31 31 100 100 100 199 199 199"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
B0="256 256 256 1024 1024 1024 2048 2048 2048 2000 2000 32"
SWEEP=""
for n in 2 3 4 5 6 8 10 12 16 20 24 28 32 40 48 56 64 80 96 112 128 160 192 224 256 320 384 448 512; do
    SWEEP="$SWEEP $n $n $n"
done
FINE=""
for n in $(seq 1 64); do
    FINE="$FINE $n $n $n"
done
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }

HWMON=
for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = coretemp ] && HWMON=$h
done
cond() {   # cond KIND VARIANT ROUND: conditions before a measurement
    t=; [ -n "$HWMON" ] && t=$(($(cat $HWMON/temp1_input)/1000))
    f=$(awk -v c=$CPU '$1=="processor"{p=$3} $1=="cpu" && $2=="MHz" && p==c {print $4}' /proc/cpuinfo)
    echo "$(date +%s),$1,$2,$3,$(cut -d' ' -f1 /proc/loadavg),$t,$f,$(who | wc -l)" >> $L/conditions.csv
}
echo "time,kind,variant,round,load1,package_temp_C,cpu3_MHz,sessions" > $L/conditions.csv

echo "start $(date -Iseconds), governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor), no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null), load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L/run.log
for r in $(seq 1 $SWEEP_ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        if [ $r -le $ROUNDS ]; then
            cond main $v $r;  taskset -c $CPU ./bench-$v 7 1 $SMALL $SQ $THIN > res/main-$v-$r.txt || fail $? "main $v $r"
            cond beta0 $v $r; taskset -c $CPU ./bench-$v 7 0 $B0 > res/beta0-$v-$r.txt || fail $? "beta0 $v $r"
        fi
        cond sweep $v $r; taskset -c $CPU ./bench-$v 7 1 $SWEEP > res/sweep-$v-$r.txt || fail $? "sweep $v $r"
        cond fine $v $r;  taskset -c $CPU ./bench-$v 7 1 $FINE > res/fine-$v-$r.txt || fail $? "fine $v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
for v in $VARIANTS; do
    taskset -c $CPU ./verify-$v 768 1024 1536 2048 3072 4000 4096 > res/verify-$v.txt 2>&1 || fail $? "verify $v"
done
echo "wrong results: $(cat res/main-* res/beta0-* res/sweep-* res/fine-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
