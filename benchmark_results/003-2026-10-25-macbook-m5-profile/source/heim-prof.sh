#!/bin/sh
#  heim-prof.sh W : breakdown and packing variants on heim with the driver
#  of prof.cc (prof-fixed.cc: 64 byte aligned buffers), SSE3 and AVX/FMA
#  micro kernel, source tree master, pinned to CPU 3, 3 rounds.
set -u
W=${1:?}; CPU=${CPU:-3}; cd $W || exit 1
TMPDIR=$W/tmp; export TMPDIR
: > failures-prof
fail() { echo "FAILED (exit $1): $2" | tee -a failures-prof; }
for a in sse avx; do
    if [ $a = sse ]; then af="-msse3 -mfpmath=sse"; else af="-mavx -mfma"; fi
    g++ -std=c++14 -O3 -fomit-frame-pointer $af -Iv-master-$a prof-fixed.cc -o prof2-$a || fail $? "prof $a"
done
for r in 1 2 3; do
    mkdir -p res-prof/r$r
    for a in sse avx; do
        for s in "256 256 256" "1024 1024 1024" "2048 2048 2048" "2048 64 2048" "64 2048 2048" "2000 2000 32"; do
            t=$(echo $s | tr ' ' -)
            taskset -c $CPU ./prof2-$a -m lib,inst,nopack,kern $s > res-prof/r$r/base-$a-$t.txt 2>&1 || fail $? "base $a $t"
            for v in ulm:ulm unit:unit blk4:unit blk8:unit; do
                taskset -c $CPU ./prof2-$a -p $v -m inst $s > res-prof/r$r/pk-$(echo $v | tr : _)-$a-$t.txt 2>&1 || fail $? "pk $v $a $t"
            done
        done
    done
    echo "round $r done $(date '+%T')"
done
if [ -s failures-prof ]; then cat failures-prof; else echo "overall: all commands succeeded"; fi
