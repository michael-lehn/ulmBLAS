#!/bin/sh
#  bench-sme.sh B : benchmark campaign of the SME2 backend on the Apple M5.
#  B is the directory of the programs (built by build-sme.sh):
#
#    bench2-V, bench3-V   dgemm and level 3 (dsyrk, dsymm, dtrmm, dtrsm)
#                         for the variants V:
#        sme        ulmBLAS feature/sme-dgemm (SME2 kernel, -mcpu=apple-m4)
#        neon       ulmBLAS master (NEON kernel)
#        blis-sme   BLIS pull request #921, configuration m4sme_p (SME2)
#        blis       BLIS 2.0, configuration firestorm (NEON)
#        openblas   OpenBLAS 0.3.34 (NEON)
#        accelerate Apple Accelerate, one thread (VECLIB_MAXIMUM_THREADS=1)
#    ukernel-sme, ukernel-blis-sme, ukernel-neon   micro kernels alone
#    smepeak                                       FMOPA peak
#
#  Every measurement is one program run under the thermal guard (guard.sh:
#  thermal state nominal before and after, otherwise repeated).  ROUNDS
#  rounds with the order of the variants rotated.  Results in B/res, one
#  file per kind of sizes, variant and round.
set -u
B=${1:?}; ROUNDS=${ROUNDS:-3}; LOG=$B/run
S=$(cd "$(dirname "$0")" && pwd); D=$B; . $S/guard.sh
mkdir -p $B/res; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
./thermal 10 > $LOG.thermal & TH=$!; trap 'kill $TH' 0
SMALL="8 8 8 16 16 16 24 24 24 32 32 32 48 48 48 64 64 64 100 100 100 128 128 128 192 192 192 256 256 256"
MID="384 384 384 512 512 512 768 768 768 1024 1024 1024"
RECT="4096 1024 1024 1024 4096 1024 1024 1024 4096 3000 2000 1000 1000 3000 2000"
THINK="2000 2000 32 2000 2000 128 4000 4000 64 1024 1024 32"
THINN="2048 64 2048 4096 96 4096 4096 16 4096 2048 256 2048"
THINM="64 2048 2048 96 4096 4096 16 4096 4096 256 2048 2048"
B0="256 256 256 1024 1024 1024 2048 2048 2048 2000 2000 32"
KCS="16 32 64 128 256 512 1024 2240 4096"
prog() {   # VARIANT PROGRAM: command line of a program of a variant
    if [ $1 = accelerate ]; then echo "env VECLIB_MAXIMUM_THREADS=1 ./$2-$1"; else echo "./$2-$1"; fi
}
echo "== start $(date '+%F %T'), rounds $ROUNDS, pause $PAUSE s, $(status)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r sme neon blis-sme blis openblas accelerate); do
        p=$(prog $v bench2); p3=$(prog $v bench3)
        measure res/small-$v-$r.txt $p 7 1 $SMALL
        measure res/mid-$v-$r.txt   $p 7 1 $MID
        for n in 1536 2048 3072 4096; do
            measure res/sq$n-$v-$r.txt $p 5 1 $n $n $n
        done
        measure res/rect-$v-$r.txt  $p 5 1 $RECT
        measure res/think-$v-$r.txt $p 7 1 $THINK
        measure res/thinn-$v-$r.txt $p 7 1 $THINN
        measure res/thinm-$v-$r.txt $p 7 1 $THINM
        measure res/beta0-$v-$r.txt $p 7 0 $B0
        for n in 512 1024 2048; do
            measure res/l3n$n-$v-$r.txt $p3 5 $n
        done
        echo "round $r $v done $(date '+%T'): $(status)"
    done
    for k in $(rot $r sme blis-sme neon); do
        measure res/ukernel-$k-$r.txt ./ukernel-$k 1 1 $KCS
    done
    measure res/smepeak-$r.txt ./smepeak
    echo "round $r micro kernels done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards), wrong results: $(cat res/*.txt | grep -c WRONG)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
echo "== done $(date '+%F %T'), $(status)"
