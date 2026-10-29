#!/bin/sh
#  heim-ab.sh W : which part of the packing change causes the slowdown of
#  small problems on heim: libraries with only the grouping of gepack_A without (pG) or only the unit stride loop without the grouping (pU)
#  or only the change of gepack_B (pB), compared with master and p4; small
#  DGEMM sizes, SSE3 and AVX/FMA, interleaved, 3 rounds, pinned to CPU
#  (default 5, not an SMT sibling of CPU 3).
set -u
W=${1:?}; CPU=${CPU:-5}; cd $W || exit 1
TMPDIR=$W/tmp; export TMPDIR
: > failures-ab22
fail() { echo "FAILED (exit $1): $2" | tee -a failures-ab22; }
echo "CPU $CPU siblings: $(cat /sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list), CPU 3 siblings: $(cat /sys/devices/system/cpu/cpu3/topology/thread_siblings_list)"
for v in pG pU; do
    for a in sse avx; do
        if [ $a = sse ]; then af="-msse3 -mfpmath=sse"; else af="-mavx -mfma"; fi
        n=$v-$a; rm -rf v-$n; cp -R master v-$n
        cp gepack-${v#p}.tcc v-$n/ulmblas/level3/pack/gepack.tcc
        (cd v-$n && make -j6 cblas ARCHFLAGS="$af") > build-$n.log 2>&1 || { fail $? "build $n"; continue; }
        gcc -O2 -Iv-$n bench2.c v-$n/libcblas.a -lstdc++ -lm -o bench2-$n || fail $? "link $n"
    done
done
SMALL="16 16 16 32 32 32 64 64 64 128 128 128 256 256 256"
for r in 1 2 3; do
    mkdir -p res-ab2/r$r
    for a in avx sse; do
        for v in master pA pG pU; do
            taskset -c $CPU ./bench2-$v-$a 7 1 $SMALL > res-ab2/r$r/small-$v-$a.txt 2>&1 || fail $? "$v $a"
        done
    done
done
if [ -s failures-ab22 ]; then cat failures-ab22; else echo "overall: all commands succeeded"; fi
