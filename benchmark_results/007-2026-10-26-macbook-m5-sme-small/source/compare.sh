#!/bin/sh
#  compare.sh B : DGEMM (column major, NoTrans, NoTrans, alpha = 1, beta =
#  1, one thread) of all programs B/bench2-V, under the same conditions as
#  the campaign 2026-10-25-macbook-m5-sme-square (bench2-accelerate with
#  VECLIB_MAXIMUM_THREADS=1).
#
#  Interleaved per shape: in every round and for every shape all programs
#  run one after the other, each in its own program run (bench2 SAMPLES 1
#  m n k: check against a naive product for small shapes, warm-up >= 0.3 s,
#  SAMPLES samples of >= 0.2 s, median, min, max), in a random order per
#  round and shape (seed 1000 * round + index of the shape); the order of
#  the shapes alternates between rounds.  Every program run under the
#  thermal guard (guard.sh).  Results: B/res/V-m_n_k-R.txt, logs B/run*.
#
#  Environment: ROUNDS (default 5), PAUSE (default 3), SAMPLES (default 5),
#  SHAPES (m_n_k ...; default below).
set -u
B=${1:?usage: packtail-bench.sh B}; ROUNDS=${ROUNDS:-5}; SAMPLES=${SAMPLES:-5}
PAUSE=${PAUSE:-3}; LOG=$B/run
S=$(cd "$(dirname "$0")" && pwd); D=$B
SHAPES=${SHAPES:-"12_12_12 12_12_16 16_12_12 12_8_12 16_16_12 20_24_20
8_8_8 9_9_9 10_10_10 13_13_13 14_14_14 15_15_15 16_16_16 17_17_17
20_20_20 23_23_23 24_24_24 28_28_28 32_32_32 36_36_36 40_40_40 64_64_64
100_100_100 250_250_250 500_500_500 1000_1000_1000 1024_1024_1024
1500_1500_1500 2047_2047_2047 4096_4096_4096"}
cd $B || exit 1
V=$(/bin/ls bench2-* | sed 's/bench2-//')
. $S/guard.sh
shuffle() { seed=$1; shift; echo "$@" | awk -v s=$seed '{ srand(s); for (i = 1; i <= NF; ++i) a[i] = $i;
    for (i = NF; i > 1; --i) { j = int(rand() * i) + 1; t = a[i]; a[i] = a[j]; a[j] = t }
    for (i = 1; i <= NF; ++i) printf "%s%s", a[i], (i < NF ? " " : "\n") }'; }
rev() { out=; for x in "$@"; do out="$x $out"; done; echo $out; }
mkdir -p res; : > $LOG.failures; : > $LOG.discards; : > $LOG.order
echo "== start $(date '+%F %T'), variants: $(echo $V), rounds $ROUNDS, pause $PAUSE s, $(status)" | tee $LOG
for r in $(seq 1 $ROUNDS); do
    if [ $((r%2)) = 1 ]; then L=$SHAPES; else L=$(rev $SHAPES); fi
    i=0
    for s in $L; do
        i=$((i+1))
        set -- $(echo ${s%s} | tr _ ' ')
        for v in $(shuffle $((1000*r+i)) $V); do
            echo "$(date +%T) $r $s $v" >> $LOG.order
            if [ $v = accelerate ]; then e="env VECLIB_MAXIMUM_THREADS=1"; else e=; fi
            measure res/$v-$s-$r.txt $e ./bench2-$v $SAMPLES 1 $1 $2 $3
            if grep -q 'thermal state' $LOG.failures; then
                echo "ABORT $(date +%T): persistent throttling" | tee -a $LOG; exit 2
            fi
        done
    done
    echo "round $r done $(date '+%T'): $(status)" | tee -a $LOG
done
echo "discarded: $(wc -l < $LOG.discards), wrong results: $(cat res/*.txt | grep -c WRONG)" | tee -a $LOG
echo "== done $(date '+%F %T'), $(status)" | tee -a $LOG
