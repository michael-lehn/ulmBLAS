#!/bin/sh
#  bench-final.sh : dgemm and level 3 benchmarks of the final source state
#  (tree /tmp/ulmblas-avx.tGdSUq/final, AVX: MC=288 KC=256 NC=4080), with SSE, BLIS and OpenBLAS;
#  below /tmp/ulmblas-avx.tGdSUq/bench-final.  Single threaded, core 3, rounds rotated.
set -u
D=/tmp/ulmblas-avx.tGdSUq; B=$D/bench-final; R=$D/refs; ROUNDS=3; CPU=3
mkdir -p $B/res; TMPDIR=$D/tmp; export TMPDIR; cd $B || exit 1; : > failures
fail() { echo "FAILED (exit $1): $2" | tee -a failures; }
for v in sse avx; do
    if [ $v = sse ]; then a="-msse3 -mfpmath=sse"; else a="-mavx -mfma"; fi
    rm -rf v-$v; cp -R $D/final v-$v
    (cd v-$v && make -j12 cblas ARCHFLAGS="$a") > build-$v.log 2>&1 || fail $? "build $v"
    echo "$v: $(grep -m1 -o -- ' -std=c++11 .*-O3 -fomit-frame-pointer' v-$v/interfaces/blas/C/build.cblas/compile.cmd)"
    gcc -O2 -Iv-$v $D/bench2.c v-$v/libcblas.a -lstdc++ -lm -o bench-$v || fail $? "link $v"
    gcc -O2 -Iv-$v $D/bench3.c v-$v/libcblas.a -lstdc++ -lm -o bench3-$v || fail $? "link3 $v"
done
cp $D/bench/bench-blis $D/bench/bench-openblas $D/bench/bench3-blis $D/bench/bench3-openblas .
SQ="64 64 64 128 128 128 192 192 192 256 256 256 384 384 384 512 512 512 768 768 768 1024 1024 1024 1536 1536 1536 2048 2048 2048 3072 3072 3072 4096 4096 4096"
SMALL="7 7 7 31 31 31 100 100 100 199 199 199"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
B0="256 256 256 1024 1024 1024 2048 2048 2048 2000 2000 32"
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor), load $(cut -d' ' -f1-3 /proc/loadavg)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r sse avx blis openblas); do
        taskset -c $CPU ./bench-$v 7 1 $SMALL $SQ $THIN > res/main-$v-$r.txt || fail $? "main $v $r"
        taskset -c $CPU ./bench-$v 7 0 $B0 > res/beta0-$v-$r.txt || fail $? "beta0 $v $r"
        taskset -c $CPU ./bench3-$v 5 512 1024 2048 3072 > res/l3-$v-$r.txt || fail $? "l3 $v $r"
        echo "round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)"
    done
done
echo "wrong results: $(cat res/*.txt | grep -c WRONG)"
if [ -s failures ]; then cat failures; else echo "overall: all commands succeeded"; fi
