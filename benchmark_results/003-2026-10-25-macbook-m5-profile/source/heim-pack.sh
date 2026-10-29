#!/bin/sh
#  heim-pack.sh W : verification of the packing change on heim, everything
#  below the workspace W (the source trees master, p4, p8 and the programs
#  bench2.c, bench3.c, prof.cc, packvar.h unpacked in W):
#  1. make check of p4 with GCC 12 (c++) and Clang 16;
#  2. CBLAS libraries of master, p4, p8 with the SSE3 and the AVX/FMA micro
#     kernel (GCC 12), benchmark programs bench2 and bench3, ROUNDS rounds,
#     the variants of each group of sizes interleaved, pinned to CPU 3.
set -u
W=${1:?}; ROUNDS=${ROUNDS:-5}; CPU=${CPU:-3}; J=12
cd $W || exit 1
TMPDIR=$W/tmp; export TMPDIR; mkdir -p tmp res
: > failures
fail() { echo "FAILED (exit $1): $2" | tee -a failures; }
summary() { grep -E 'PASS [0-9]+, FAIL' $1 | tail -1; }

echo "== checks start $(date '+%F %T')"
for c in gcc clang; do
    rm -rf chk-$c; cp -R p4 chk-$c
    if [ $c = gcc ]; then mk=""; else mk="CXX=clang++-16 CC=clang-16"; fi
    (cd chk-$c && make -j$J check $mk) > chk-$c.log 2>&1 || fail $? "make check $c"
    echo "$c: $(summary chk-$c.log), warnings $(grep -c -i 'warning' chk-$c.log), jobserver $(grep -c -i 'jobserver' chk-$c.log)"
done

echo "== builds $(date '+%T')"
for v in master p4 p8; do
    for a in sse avx; do
        if [ $a = sse ]; then af="-msse3 -mfpmath=sse"; else af="-mavx -mfma"; fi
        n=$v-$a; rm -rf v-$n; cp -R $v v-$n
        (cd v-$n && make -j$J cblas ARCHFLAGS="$af") > build-$n.log 2>&1 || { fail $? "build $n"; continue; }
        for p in bench2 bench3; do
            gcc -O2 -Iv-$n $p.c v-$n/libcblas.a -lstdc++ -lm -o $p-$n || fail $? "link $p $n"
        done
    done
done
for a in sse avx; do
    if [ $a = sse ]; then af="-msse3 -mfpmath=sse"; else af="-mavx -mfma"; fi
    for v in master p4; do
        g++ -std=c++14 -O3 -fomit-frame-pointer $af -Iv-$v-$a prof.cc -o prof-$v-$a || fail $? "prof $v $a"
    done
done

SMALL="16 16 16 32 32 32 64 64 64 100 100 100 128 128 128 192 192 192 199 199 199 256 256 256"
MID="384 384 384 512 512 512 768 768 768 1024 1024 1024"
THIN="2000 2000 32 2000 2000 128 4000 4000 64 2048 64 2048 64 2048 2048 4096 96 4096 96 4096 4096"
echo "== benchmarks start $(date '+%T'), $ROUNDS rounds, CPU $CPU"
for r in $(seq 1 $ROUNDS); do
    mkdir -p res/r$r
    for a in sse avx; do
        for g in small mid sq2048 sq4096 thin l3n512 l3n1024 l3n2048; do
            case $r in 1) V="master p4 p8";; 2) V="p4 p8 master";; 3) V="p8 master p4";;
                        4) V="master p8 p4";; *) V="p4 master p8";; esac
            for v in $V; do
                n=$v-$a
                case $g in
                    small)  cmd="./bench2-$n 7 1 $SMALL" ;;
                    mid)    cmd="./bench2-$n 7 1 $MID" ;;
                    sq2048) cmd="./bench2-$n 5 1 2048 2048 2048" ;;
                    sq4096) cmd="./bench2-$n 5 1 4096 4096 4096" ;;
                    thin)   cmd="./bench2-$n 7 1 $THIN" ;;
                    l3n*)   cmd="./bench3-$n 5 ${g#l3n}" ;;
                esac
                taskset -c $CPU $cmd > res/r$r/$g-$n.txt 2>&1 || fail $? "$g $n round $r"
            done
        done
        #  breakdown (driver of prof.cc) for master and p4
        for v in master p4; do
            for s in "1024 1024 1024" "2048 64 2048" "256 256 256"; do
                taskset -c $CPU ./prof-$v-$a -m lib,inst $s >> res/r$r/prof-$v-$a.txt 2>&1 || fail $? "prof $v $a"
            done
        done
    done
    echo "round $r done $(date '+%T'), load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "wrong results: $(cat res/r*/*.txt | grep -c WRONG)"
if [ -s failures ]; then cat failures; else echo "overall: all commands succeeded"; fi
echo "== done $(date '+%F %T')"
