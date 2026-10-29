#!/bin/sh
#  sweep-blocks.sh OUT PROF : block sizes of the SME2 configuration.  PROF
#  is the profiling driver (prof.cc) built against the ulmBLAS tree; its
#  -b option sets MC,KC,NC at run time (the packing routines and the micro
#  kernel are those of the library).  Stage 1: MC x KC with NC = 3072;
#  stage 2: NC with the MC, KC given in NC_MC, NC_KC.  Each measurement
#  under the thermal guard (guard.sh); output OUT/<stage>.txt, one line per
#  run (mode inst: the instrumented driver with the given block sizes).
set -u
OUT=${1:?}; PROF=${2:?}
D=${D:?directory of the programs peak and thermal}; LOG=$OUT/log
mkdir -p $OUT; PAUSE=${PAUSE:-5}
. "$(dirname "$0")/guard.sh"
SHAPES="1024,1024,1024 4096,4096,4096 4096,96,4096 2000,2000,128 64,2048,2048"
echo "start $(date '+%F %T'), $(status)" >> $LOG
: > $OUT/stage1.txt
for kc in 512 1024 2240 4096; do
    for mc in 128 256 384 608 1024; do
        for s in $SHAPES; do
            set -- $(echo $s | tr , ' ')
            measure $OUT/run.txt $PROF -s 3 -m inst -b $mc,$kc,3072 $1 $2 $3 \
                && cat $OUT/run.txt >> $OUT/stage1.txt
        done
    done
done
echo "stage 1 done $(date '+%T'), $(status)" >> $LOG
: > $OUT/stage2.txt
for nc in 1024 2048 3072 4096 8192; do
    for s in $SHAPES; do
        set -- $(echo $s | tr , ' ')
        measure $OUT/run.txt $PROF -s 3 -m inst -b ${NC_MC:-608},${NC_KC:-2240},$nc \
            $1 $2 $3 && cat $OUT/run.txt >> $OUT/stage2.txt
    done
done
echo "done $(date '+%F %T'), $(status)" >> $LOG
