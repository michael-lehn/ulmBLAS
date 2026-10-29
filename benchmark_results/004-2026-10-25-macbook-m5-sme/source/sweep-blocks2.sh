#!/bin/sh
#  sweep-blocks2.sh OUT PROF : focused sweep of MC x KC x NC for the SME2
#  configuration (after sweep-blocks.sh); each measurement under the
#  thermal guard, output OUT/sweep.txt (prof mode inst).
set -u
OUT=${1:?}; PROF=${2:?}
D=${D:?directory of the programs peak and thermal}; LOG=$OUT/log
mkdir -p $OUT; PAUSE=${PAUSE:-5}
. "$(dirname "$0")/guard.sh"
SHAPES="512,512,512 1024,1024,1024 4096,4096,4096 4096,96,4096 2000,2000,128 64,2048,2048"
echo "start $(date '+%F %T'), $(status)" >> $LOG
: > $OUT/sweep.txt
for nc in 1024 2048 3072; do
    for kc in 1024 1536 2240; do
        for mc in 256 384 608; do
            for s in $SHAPES; do
                set -- $(echo $s | tr , ' ')
                measure $OUT/run.txt $PROF -s 3 -m inst -b $mc,$kc,$nc $1 $2 $3 \
                    && cat $OUT/run.txt >> $OUT/sweep.txt
            done
        done
    done
    echo "nc $nc done $(date '+%T'), $(status)" >> $LOG
done
echo "done $(date '+%F %T'), $(status)" >> $LOG
