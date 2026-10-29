#!/bin/sh
#  ab-blocks.sh B : interleaved A/B of the block sizes in the library:
#  bench2-v5 (889aa15, MC, KC, NC = 608, 2240, 3072) against bench2-final
#  (8fe961a, 384, 1024, 1024), the same micro kernel and packing.  ROUNDS
#  rounds, order alternating, each run under the thermal guard (guard.sh);
#  results in B/res/V-R.txt.
set -u
B=${1:?}; ROUNDS=${ROUNDS:-4}; LOG=$B/run
S=$(cd "$(dirname "$0")" && pwd); D=$B; . $S/guard.sh
mkdir -p $B/res; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
SIZES="1024 1024 1024 2048 2048 2048 3072 3072 3072 4096 4096 4096 4096 96 4096 64 2048 2048 2000 2000 128 512 512 512"
echo "== start $(date '+%F %T'), $(status)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r v5 final); do
        measure res/$v-$r.txt ./bench2-$v 5 1 $SIZES
    done
    echo "round $r done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
echo "== done $(date '+%F %T'), $(status)"
