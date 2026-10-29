#!/bin/sh
#  study-ref.sh B : the reference curves of the block size study on the same
#  shapes as study-shapes.sh: B/bench2-accelerate (Apple Accelerate, one
#  thread), B/bench2-neon (ulmBLAS NEON) and two SME2 configurations of the
#  study as anchors (B/bench2-b384_1024_1024, B/bench2-b608_2240_3072), so
#  that the difference between the two sessions can be measured.  ROUNDS
#  rounds, order rotated, thermal guard; results in B/res/V-R.txt.
set -u
B=${1:?}; ROUNDS=${ROUNDS:-3}; LOG=$B/run
S=$(cd "$(dirname "$0")" && pwd); D=$B; . $S/guard.sh
mkdir -p $B/res; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
SQUARE=""; for n in 256 384 512 768 1024 1536 2048 3072 4096; do SQUARE="$SQUARE $n $n $n"; done
SMALLM=""; for m in 16 32 64 96 128 192 256 384; do SMALLM="$SMALLM $m 2048 2048"; done
SMALLN=""; for n in 16 32 64 96 128 256; do SMALLN="$SMALLN 2048 $n 2048"; done
SMALLK=""; for k in 32 64 128 256 512; do SMALLK="$SMALLK 2048 2048 $k"; done
V="accelerate neon b384_1024_1024 b608_2240_3072"
echo "== start $(date '+%F %T'), variants $V, $(status)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $V); do
        if [ $v = accelerate ]; then p="env VECLIB_MAXIMUM_THREADS=1 ./bench2-$v"; else p=./bench2-$v; fi
        measure res/$v-$r.txt $p 5 1 $SQUARE $SMALLM $SMALLN $SMALLK
    done
    echo "round $r done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
echo "== done $(date '+%F %T'), $(status)"
