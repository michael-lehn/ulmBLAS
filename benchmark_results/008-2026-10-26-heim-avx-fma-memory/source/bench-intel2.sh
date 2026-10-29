#!/bin/sh
#  bench-intel2.sh WORKSPACE : DGEMM and level 3 benchmarks on heim
#  (Intel Core i5-12500), single threaded, core 3, with the methodology of
#  campaign 2026-10-24-heim-avx-fma (run B): bench2.c / bench3.c, 7 (main,
#  beta0) or 5 (level 3) samples of at least 0.2 s, median; rounds with
#  rotated order of the variants.  Variants:
#
#    avx-pub   ulmBLAS e195bea (published run B), AVX/FMA
#    avx-pre   ulmBLAS 57e1c8a (before the per-thread buffer cache), AVX/FMA
#    avx-cache ulmBLAS 04cf17b (per-thread buffer cache), AVX/FMA
#    avx-cur   ulmBLAS febb852 (current master), AVX/FMA
#    sse-pub   ulmBLAS e195bea, SSE
#    sse-cur   ulmBLAS febb852, SSE
#    blis      BLIS 2.0 (library and program of run B)
#    openblas  OpenBLAS 0.3.34 (library and program of run B)
#
set -u
W=$1; cd $W || exit 1
D=/tmp/ulmblas-avx.tGdSUq          # workspace of campaign 2026-10-24
B=$W/bench; L=$W/logs; mkdir -p $B/res $L; cd $B || exit 1
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
CPU=3; ROUNDS=3; SWEEP_ROUNDS=5
VARIANTS="avx-pub avx-pre avx-cache avx-cur sse-pub sse-cur blis openblas"

rev() { case $1 in *-pub) echo e195bea;; avx-pre) echo 57e1c8a;; avx-cache) echo 04cf17b;; *-cur) echo febb852;; esac; }

for v in avx-pub avx-pre avx-cache avx-cur sse-pub sse-cur; do
    r=`rev $v`
    case $v in avx-*) a="-mavx -mfma";; *) a="-msse3 -mfpmath=sse";; esac
    rm -rf v-$v; cp -R $W/$r v-$v
    (cd v-$v && make -j12 cblas ARCHFLAGS="$a") > $L/build-$v.log 2>&1 || fail $? "build $v"
    echo "$v ($r): $(cat v-$v/interfaces/blas/C/build.cblas/compile.cmd)" >> $L/compile-commands.txt
    gcc -O2 -Iv-$v $D/bench2.c v-$v/libcblas.a -lstdc++ -lm -o bench-$v || fail $? "link $v"
    gcc -O2 -Iv-$v $D/bench3.c v-$v/libcblas.a -lstdc++ -lm -o bench3-$v || fail $? "link3 $v"
    gcc -O2 -Iv-$v $W/verify-large.c v-$v/libcblas.a -lstdc++ -lm -o verify-$v || fail $? "verify $v"
done
for v in blis openblas; do
    cp $D/bench/bench-$v $D/bench/bench3-$v .
done
gcc -O2 -I$D/refs/blis/include/blis $W/verify-large.c $D/refs/blis/lib/libblis.a -lm -lpthread -o verify-blis || fail $? "verify blis"
gcc -O2 -I$D/refs/openblas/include $W/verify-large.c $D/refs/openblas/lib/libopenblas.a -lm -lpthread -o verify-openblas || fail $? "verify openblas"
sha256sum bench-* bench3-* verify-* > $L/programs.sha256

SQ="64 64 64 128 128 128 192 192 192 256 256 256 384 384 384 512 512 512 768 768 768 1024 1024 1024 1536 1536 1536 2048 2048 2048 3072 3072 3072 4096 4096 4096"
SMALL="7 7 7 31 31 31 100 100 100 199 199 199"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
B0="256 256 256 1024 1024 1024 2048 2048 2048 2000 2000 32"
SWEEP=""
for n in 2 3 4 5 6 8 10 12 16 20 24 28 32 40 48 56 64 80 96 112 128 160 192 224 256 320 384 448 512; do
    SWEEP="$SWEEP $n $n $n"
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

echo "start $(date -Iseconds), governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor), load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L/run.log
for r in $(seq 1 $SWEEP_ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        if [ $r -le $ROUNDS ]; then
            cond main $v $r;  taskset -c $CPU ./bench-$v 7 1 $SMALL $SQ $THIN > res/main-$v-$r.txt || fail $? "main $v $r"
            cond beta0 $v $r; taskset -c $CPU ./bench-$v 7 0 $B0 > res/beta0-$v-$r.txt || fail $? "beta0 $v $r"
            cond l3 $v $r;    taskset -c $CPU ./bench3-$v 5 512 1024 2048 3072 > res/l3-$v-$r.txt || fail $? "l3 $v $r"
        fi
        cond sweep $v $r; taskset -c $CPU ./bench-$v 7 1 $SWEEP > res/sweep-$v-$r.txt || fail $? "sweep $v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
for v in $VARIANTS; do
    taskset -c $CPU ./verify-$v 768 1024 1536 2048 3072 4000 4096 > res/verify-$v.txt 2>&1 || fail $? "verify $v"
done
echo "wrong results: $(cat res/main-* res/beta0-* res/sweep-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
