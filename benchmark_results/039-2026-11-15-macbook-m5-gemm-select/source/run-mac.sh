#!/bin/bash
#  MacBook M5: master vs redesign (hand-written table, ULMBLAS_PLATFORM=
#  generic) vs redesign (tuned table), types d s c z, ROUNDS rounds; each
#  variant x type a program run under the thermal and load guard, order
#  rotated per round.  Output: data/<variant>-<type>-<round>.tsv
S=$(cd "$(dirname "$0")" && pwd)
D=$S/bin; LOG=$S/guard.log; LOADMAX=${LOADMAX:-2.5}; PAUSE=${PAUSE:-10}
ROUNDS=${ROUNDS:-5}
. $HOME/git/ulmBLAS/benchmark_results/022-2026-10-27-macbook-m5-sme-blocking/source/guard.sh
mkdir -p $S/data $S/bin $S/master
#  master: the library of origin/master
if [ ! -d $S/master/ulmblas ]; then
    git -C $HOME/git/ulmBLAS archive origin/master ulmblas | tar x -C $S/master
fi
cc -O2 -framework Foundation $HOME/git/ulmBLAS/benchmark_results/022-2026-10-27-macbook-m5-sme-blocking/source/thermal.m -o $S/bin/thermal
F="-std=c++11 -O3 -mcpu=apple-m4"
clang++ $F -I$S/master $S/selbench.cc -o $S/bin/sel-master || exit 1
clang++ $F -I$HOME/git/ulmBLAS-gemm-redesign $S/selbench.cc -o $S/bin/sel-redesign || exit 1
variants="master generic tuned"
for r in $(seq 1 $ROUNDS); do
    order=$(rot $r $variants)
    for t in d s c z; do
        for v in $order; do
            case $v in
                master)  cmd="$S/bin/sel-master $t" ;;
                generic) cmd="env ULMBLAS_PLATFORM=generic $S/bin/sel-redesign $t" ;;
                tuned)   cmd="env ULMBLAS_PLATFORM=apple-m5 $S/bin/sel-redesign $t" ;;
            esac
            measure $S/data/$v-$t-$r.tsv $cmd || exit 1
        done
    done
    echo "round $r done $(date +%T) $(uptime | sed "s/.*load/load/")"
done
echo done
