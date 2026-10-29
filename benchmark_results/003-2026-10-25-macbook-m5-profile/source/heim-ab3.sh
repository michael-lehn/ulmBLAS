#!/bin/sh
#  heim-ab3.sh W : where does the grouping of panels in gepack_A (pG) start
#  to pay off on heim: master and pG (and pU: only the separate loop for
#  unit row stride), DGEMM n = 192 ... 2048 and thin shapes, SSE3 and
#  AVX/FMA, interleaved, 3 rounds, pinned to CPU 5.
set -u
W=${1:?}; CPU=${CPU:-5}; cd $W || exit 1
TMPDIR=$W/tmp; export TMPDIR
: > failures-ab3
fail() { echo "FAILED (exit $1): $2" | tee -a failures-ab3; }
SQ="192 192 192 256 256 256 384 384 384 512 512 512 768 768 768 1024 1024 1024 1536 1536 1536 2048 2048 2048"
THIN="2048 64 2048 4096 96 4096 1024 64 1024 512 64 512 1024 32 1024 2000 2000 32"
for r in 1 2 3; do
    mkdir -p res-ab3/r$r
    for a in avx sse; do
        for g in sq thin; do
            if [ $g = sq ]; then S=$SQ; else S=$THIN; fi
            for v in master pG pU; do
                taskset -c $CPU ./bench2-$v-$a 5 1 $S > res-ab3/r$r/$g-$v-$a.txt 2>&1 || fail $? "$g $v $a"
            done
        done
    done
done
if [ -s failures-ab3 ]; then cat failures-ab3; else echo "overall: all commands succeeded"; fi
