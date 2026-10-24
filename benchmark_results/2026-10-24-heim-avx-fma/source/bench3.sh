#!/bin/sh
#  bench3.sh : level 3 (syrk, symm, trmm, trsm) benchmarks, below /tmp/ulmblas-avx.tGdSUq/bench
set -u
D=/tmp/ulmblas-avx.tGdSUq; B=$D/bench; R=$D/refs; CPU=3; ROUNDS=3
cd $B || exit 1; TMPDIR=$D/tmp; export TMPDIR; : > failures3
fail() { echo "FAILED (exit $1): $2" | tee -a failures3; }
for v in sse avx-72-256 avx-144-256 avx-288-256; do
    gcc -O2 -Iv-$v $D/bench3.c v-$v/libcblas.a -lstdc++ -lm -o bench3-$v || fail $? "link $v"
done
gcc -O2 -I$R/blis/include/blis $D/bench3.c $R/blis/lib/libblis.a -lm -lpthread -o bench3-blis || fail $? "link blis"
gcc -O2 -I$R/openblas/include $D/bench3.c $R/openblas/lib/libopenblas.a -lm -lpthread -o bench3-openblas || fail $? "link openblas"
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r sse avx-72-256 avx-144-256 avx-288-256 blis openblas); do
        taskset -c $CPU ./bench3-$v 5 512 1024 2048 3072 > res/l3-$v-$r.txt || fail $? "l3 $v $r"
    done
    echo "round $r done, load $(cut -d' ' -f1 /proc/loadavg)"
done
if [ -s failures3 ]; then cat failures3; else echo "overall: all commands succeeded"; fi
