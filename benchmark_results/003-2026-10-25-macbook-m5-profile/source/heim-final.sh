#!/bin/sh
#  heim-final.sh W : final verification of the packing change on heim:
#  1. make check of the final source tree (W/final: gepack_A packs 4 panels
#     together if the columns of A are at least 8 KiB apart, new test
#     T0347) with GCC 12 (c++) and Clang 16;
#  2. CBLAS libraries pT (= final) and pTG (without the separate loop for
#     unit row stride), SSE3 and AVX/FMA, compared with master (built by
#     heim-pack.sh): DGEMM and level 3, 3 rounds, interleaved, pinned to
#     CPU 3.
set -u
W=${1:?}; CPU=${CPU:-3}; J=12; cd $W || exit 1
TMPDIR=$W/tmp; export TMPDIR
: > failures-final
fail() { echo "FAILED (exit $1): $2" | tee -a failures-final; }
summary() { grep -E 'PASS [0-9]+, FAIL' $1 | tail -1; }
echo "== checks start $(date '+%F %T')"
for c in gcc clang; do
    cp -R final fchk-$c
    if [ $c = gcc ]; then mk=""; else mk="CXX=clang++-16 CC=clang-16"; fi
    (cd fchk-$c && make -j$J check $mk) > fchk-$c.log 2>&1 || fail $? "make check $c"
    echo "$c: $(summary fchk-$c.log), warnings $(grep -c -i 'warning' fchk-$c.log), jobserver $(grep -c -i 'jobserver' fchk-$c.log), T0347 $(grep -c 'T0347' fchk-$c/test/build/check.log 2>/dev/null)"
done
echo "== builds $(date '+%T')"
for v in pT pTG; do
    for a in sse avx; do
        if [ $a = sse ]; then af="-msse3 -mfpmath=sse"; else af="-mavx -mfma"; fi
        n=$v-$a; cp -R master v-$n
        cp gepack-${v#p}.tcc v-$n/ulmblas/level3/pack/gepack.tcc
        (cd v-$n && make -j$J cblas ARCHFLAGS="$af") > build-$n.log 2>&1 || { fail $? "build $n"; continue; }
        for p in bench2 bench3; do
            gcc -O2 -Iv-$n $p.c v-$n/libcblas.a -lstdc++ -lm -o $p-$n || fail $? "link $p $n"
        done
    done
done
cmp final/ulmblas/level3/pack/gepack.tcc gepack-T.tcc && echo "pT = final"
SMALL="16 16 16 32 32 32 64 64 64 100 100 100 128 128 128 192 192 192 199 199 199 256 256 256"
MID="384 384 384 512 512 512 768 768 768 1024 1024 1024"
THIN="2000 2000 32 2000 2000 128 4000 4000 64 2048 64 2048 64 2048 2048 4096 96 4096 96 4096 4096 1024 64 1024 1024 32 1024 512 64 512"
echo "== benchmarks start $(date '+%T'), CPU $CPU"
for r in 1 2 3; do
    mkdir -p res-final/r$r
    for a in sse avx; do
        for g in small mid sq2048 sq4096 thin l3n512 l3n1024 l3n2048; do
            case $r in 1) V="master pT pTG";; 2) V="pT pTG master";; *) V="pTG master pT";; esac
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
                taskset -c $CPU $cmd > res-final/r$r/$g-$n.txt 2>&1 || fail $? "$g $n round $r"
            done
        done
    done
    echo "round $r done $(date '+%T'), load $(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "wrong results: $(cat res-final/r*/*.txt | grep -c WRONG)"
if [ -s failures-final ]; then cat failures-final; else echo "overall: all commands succeeded"; fi
echo "== done $(date '+%F %T')"
