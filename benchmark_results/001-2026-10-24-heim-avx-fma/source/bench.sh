#!/bin/sh
#  bench.sh : dgemm benchmarks on heim, everything below /tmp/ulmblas-avx.tGdSUq/bench.
#  Variants: ulmBLAS ref / sse / avx (and a block size sweep of avx), BLIS 2.0,
#  OpenBLAS 0.3.34.  Single threaded, pinned to one core, ROUNDS rounds with
#  the order of the variants rotated.
set -u
D=/tmp/ulmblas-avx.tGdSUq; B=$D/bench; R=$D/refs; ROUNDS=${ROUNDS:-3}; CPU=${CPU:-3}
mkdir -p $B/res $D/tmp; TMPDIR=$D/tmp; export TMPDIR; cd $B || exit 1
: > failures
fail() { echo "FAILED (exit $1): $2" | tee -a failures; }

#  ulmBLAS variants: tree copies of stage3 with ARCHFLAGS and block sizes
mkvariant() {   # NAME ARCHFLAGS [MC KC]
    n=$1; a=$2
    rm -rf v-$n; cp -R $D/stage3 v-$n
    if [ $# -ge 4 ]; then
        sed -i -e '/#elif defined(USE_AVX)/,/#elif defined(USE_SSE)/{s/std::is_same<double, T>::value               ? 72$/std::is_same<double, T>::value               ? '$3'/;s/std::is_same<double, T>::value               ? 256$/std::is_same<double, T>::value               ? '$4'/}' v-$n/ulmblas/config/blocksize.h
    fi
    (cd v-$n && make -j12 cblas ARCHFLAGS="$a") > build-$n.log 2>&1 \
        || { fail $? "build $n"; return; }
    gcc -O2 -Iv-$n $D/bench2.c v-$n/libcblas.a -lstdc++ -lm -o bench-$n \
        || fail $? "link $n"
    echo "$n: ARCHFLAGS='$a' $(grep -A1 -E 'is_same<double, T>::value +\? [0-9]+$' v-$n/ulmblas/config/blocksize.h | head -0)"
}
mkvariant ref ""
mkvariant sse "-msse3 -mfpmath=sse"
mkvariant avx "-mavx -mfma"
for mc in 72 144 288; do for kc in 256 384 512; do
    mkvariant avx-$mc-$kc "-mavx -mfma" $mc $kc
done; done
gcc -O2 -I$R/blis/include/blis $D/bench2.c $R/blis/lib/libblis.a -lm -lpthread -o bench-blis || fail $? "link blis"
gcc -O2 -I$R/openblas/include $D/bench2.c $R/openblas/lib/libopenblas.a -lm -lpthread -o bench-openblas || fail $? "link openblas"
for n in avx-72-256 avx-288-512; do
    echo "$n block sizes: $(sed -n '/#elif defined(USE_AVX)/,/#elif defined(USE_SSE)/p' v-$n/ulmblas/config/blocksize.h | grep 'is_same<double' | awk '{print $NF}' | tr '\n' ' ')"
done

SQ="64 64 64 128 128 128 192 192 192 256 256 256 384 384 384 512 512 512 768 768 768 1024 1024 1024 1536 1536 1536 2048 2048 2048 3072 3072 3072 4096 4096 4096"
SMALL="7 7 7 31 31 31 100 100 100 199 199 199"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
SQREF="64 64 64 128 128 128 256 256 256 512 512 512 1024 1024 1024"
B0="256 256 256 1024 1024 1024 2048 2048 2048 2000 2000 32"
SWEEP="512 512 512 1024 1024 1024 2048 2048 2048 4096 4096 4096 2000 2000 256"

echo "== governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor), cpu $CPU, rounds $ROUNDS, load $(cut -d' ' -f1-3 /proc/loadavg)"
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r ref sse avx blis openblas); do
        if [ $v = ref ]; then sizes="$SMALL $SQREF"; else sizes="$SMALL $SQ $THIN"; fi
        taskset -c $CPU ./bench-$v 7 1 $sizes > res/main-$v-$r.txt || fail $? "main $v $r"
        taskset -c $CPU ./bench-$v 7 0 $B0 > res/beta0-$v-$r.txt || fail $? "beta0 $v $r"
        echo "round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)"
    done
    for v in $(rot $r avx-72-256 avx-72-384 avx-72-512 avx-144-256 avx-144-384 avx-144-512 avx-288-256 avx-288-384 avx-288-512); do
        taskset -c $CPU ./bench-$v 5 1 $SWEEP > res/sweep-$v-$r.txt || fail $? "sweep $v $r"
    done
    echo "round $r sweep done"
done
echo "wrong results: $(cat res/*.txt | grep -c WRONG)"
if [ -s failures ]; then echo "overall: $(wc -l < failures) failed"; cat failures; else echo "overall: all commands succeeded"; fi
