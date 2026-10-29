#!/bin/sh
#  study-shapes.sh B : DGEMM (bench2, beta = 1) of all programs
#  B/bench2-b* for square, small m, small n and small k shapes; ROUNDS
#  rounds, the order of the variants rotated, each program run under the
#  thermal guard (guard.sh, pause PAUSE).  Results in B/res/V-R.txt.
set -u
B=${1:?}; ROUNDS=${ROUNDS:-3}; LOG=$B/run
S=$(cd "$(dirname "$0")" && pwd); D=$B; . $S/guard.sh
mkdir -p $B/res; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
SQUARE=""; for n in 256 384 512 768 1024 1536 2048 3072 4096; do SQUARE="$SQUARE $n $n $n"; done
SMALLM=""; for m in 16 32 64 96 128 192 256 384; do SMALLM="$SMALLM $m 2048 2048"; done
SMALLN=""; for n in 16 32 64 96 128 256; do SMALLN="$SMALLN 2048 $n 2048"; done
SMALLK=""; for k in 32 64 128 256 512; do SMALLK="$SMALLK 2048 2048 $k"; done
V=$(ls bench2-b* | sed 's/bench2-//')
echo "== start $(date '+%F %T'), variants $(echo $V), $(status)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $V); do
        measure res/$v-$r.txt ./bench2-$v 5 1 $SQUARE $SMALLM $SMALLN $SMALLK
    done
    echo "round $r done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
echo "== done $(date '+%F %T'), $(status)"
