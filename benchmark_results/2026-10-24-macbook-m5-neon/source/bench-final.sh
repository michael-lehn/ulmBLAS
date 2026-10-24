#!/bin/sh
#  bench-final.sh DIR : run B, dgemm and level 3 benchmarks of the final
#  source state with the thermal guard (guard.sh).  Variants (programs built
#  by build-variants.sh in DIR/bench): base (ulmBLAS before the NEON
#  kernel), ref (reference kernels, -DULMBLAS_NO_NEON), neon (final block
#  sizes, Apple clang), neon-gcc (GCC 16), blis, openblas, accelerate (Apple
#  Accelerate, one thread: VECLIB_MAXIMUM_THREADS=1, for context).  Results
#  in DIR/bench/res-B, one file per kind of sizes, variant and round.
#  Single threaded (QoS user-interactive), ROUNDS rounds with the order of
#  the variants rotated.
set -u
D=${1:?}; B=$D/bench; ROUNDS=${ROUNDS:-3}; LOG=$B/run-B
S=$(cd "$(dirname "$0")" && pwd); . $S/guard.sh
mkdir -p $B/res-B; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
$D/thermal 10 > $LOG.thermal & TH=$!; trap 'kill $TH' 0
SMALL="7 7 7 31 31 31 64 64 64 100 100 100 128 128 128 192 192 192 199 199 199 256 256 256"
MID="384 384 384 512 512 512 768 768 768 1024 1024 1024"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
B0="256 256 256 1024 1024 1024 2048 2048 2048 2000 2000 32"
prog() {   # VARIANT PROGRAM: command line of a program of a variant
    if [ $1 = accelerate ]; then echo "env VECLIB_MAXIMUM_THREADS=1 ./$2-$1"; else echo "./$2-$1"; fi
}
echo "== run B start $(date '+%F %T'), rounds $ROUNDS, pause $PAUSE s, $(status)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r base ref neon neon-gcc blis openblas accelerate); do
        p=$(prog $v bench); p3=$(prog $v bench3)
        measure res-B/small-$v-$r.txt $p 7 1 $SMALL
        measure res-B/mid-$v-$r.txt   $p 7 1 $MID
        [ $v = ref ] && continue        # the reference kernel only up to n = 1024
        for n in 1536 2048 3072 4096; do
            measure res-B/sq$n-$v-$r.txt $p 7 1 $n $n $n
        done
        measure res-B/thin-$v-$r.txt  $p 7 1 $THIN
        measure res-B/beta0-$v-$r.txt $p 7 0 $B0
        for n in 512 1024 2048 3072; do
            measure res-B/l3n$n-$v-$r.txt $p3 5 $n
        done
        echo "round $r $v done $(date '+%T'): $(status)"
    done
done
echo "discarded: $(wc -l < $LOG.discards), wrong results: $(cat res-B/*.txt | grep -c WRONG)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
