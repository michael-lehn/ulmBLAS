#!/bin/sh
#  ab-blocks2.sh B : interleaved comparison of block sizes in the library
#  (8fe961a with MC, KC, NC of double replaced; programs B/bench2-bMC_KC_NC).
#  ROUNDS rounds, order rotated, each run under the thermal guard; results
#  in B/res/V-R.txt.
set -u
B=${1:?}; ROUNDS=${ROUNDS:-3}; LOG=$B/run
S=$(cd "$(dirname "$0")" && pwd); D=$B; . $S/guard.sh
mkdir -p $B/res; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
SIZES="512 512 512 1024 1024 1024 2048 2048 2048 3072 3072 3072 4096 4096 4096 4096 96 4096 2048 64 2048 2000 2000 128 64 2048 2048 16 4096 4096 256 2048 2048 3000 2000 1000"
V=$(cd $B && ls bench2-b* | sed 's/bench2-//')
echo "== start $(date '+%F %T'), variants $(echo $V), $(status)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $V); do
        measure res/$v-$r.txt ./bench2-$v 5 1 $SIZES
    done
    echo "round $r done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
echo "== done $(date '+%F %T'), $(status)"
