#!/bin/sh
#  ab-runlength.sh B : the same program (bench2-sme of the campaign) with
#  4096 x 4096 x 4096 alone in a program run ("alone") and after the other
#  sizes of ab-blocks2.sh in the same run ("after"), ROUNDS interleaved
#  rounds under the thermal guard; results in B/res/MODE-R.txt.
set -u
B=${1:?}; ROUNDS=${ROUNDS:-3}; LOG=$B/run
S=$(cd "$(dirname "$0")" && pwd); D=$B; . $S/guard.sh
mkdir -p $B/res; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
PRE="512 512 512 1024 1024 1024 2048 2048 2048 3072 3072 3072"
echo "== start $(date '+%F %T'), $(status)"
for r in $(seq 1 $ROUNDS); do
    for m in $(rot $r alone after); do
        if [ $m = alone ]; then
            measure res/$m-$r.txt ./bench2-sme 5 1 4096 4096 4096 2048 2048 2048
        else
            measure res/$m-$r.txt ./bench2-sme 5 1 $PRE 4096 4096 4096
        fi
    done
    echo "round $r done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards)"
echo "== done $(date '+%F %T'), $(status)"
