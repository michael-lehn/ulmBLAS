#!/bin/sh
#  square-interleaved.sh B : square DGEMM (m = n = k, column major,
#  alpha = 1, beta = 1, one thread) of all implementations in B:
#
#    bench2-accelerate   Apple Accelerate (VECLIB_MAXIMUM_THREADS=1)
#    bench2-blis-sme     BLIS pull request #921, configuration m4sme_p
#    bench2-neon         ulmBLAS NEON (default configuration)
#    bench2-bMC_KC_NC    ulmBLAS SME2 with fixed block sizes MC, KC, NC
#
#  Interleaved per size: in every round and for every size all
#  implementations run one after the other, each in its own program run
#  (bench2 SAMPLES 1 n n n: a check against a naive product for small n,
#  warm-up >= 0.3 s, SAMPLES samples of >= 0.2 s, median, min, max).  The
#  order of the implementations is a random permutation for every round and
#  size (seed 1000 * round + n, reproducible; so that no implementation
#  always follows the same other one), the order of the sizes alternates
#  between ascending (odd rounds) and descending (even rounds).  Every program run is under the thermal guard (guard.sh: pause
#  PAUSE seconds, thermal state nominal before and after, otherwise
#  repeated).  Results: B/res/V-n-R.txt; log B/run, B/run.discards,
#  B/run.failures; the order of the runs B/run.order.
#
#  Environment: ROUNDS (default 5), PAUSE (default 3), SAMPLES (default 5),
#  SIZES, DRY_RUN=1 (print the plan and the number of runs, run nothing).
#
#  Persistent throttling stops the campaign instead of collecting
#  unreliable results: it aborts (exit 2) if a run fails because the
#  thermal state was not nominal in TRIES attempts, or if more than
#  MAX_DISCARD_PERCENT (default 10) percent of the runs so far had to be
#  discarded (checked after at least 20 runs).
set -u
B=${1:?usage: square-interleaved.sh B}; ROUNDS=${ROUNDS:-5}; SAMPLES=${SAMPLES:-5}
PAUSE=${PAUSE:-3}; LOG=$B/run
SIZES=${SIZES:-"8 16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096"}
S=$(cd "$(dirname "$0")" && pwd); D=$B
cd $B || exit 1
V=$(ls bench2-* | sed 's/bench2-//')
nv=$(echo $V | wc -w); ns=$(echo $SIZES | wc -w)
shuffle() { seed=$1; shift; echo "$@" | awk -v s=$seed '{ srand(s); for (i = 1; i <= NF; ++i) a[i] = $i;
    for (i = NF; i > 1; --i) { j = int(rand() * i) + 1; t = a[i]; a[i] = a[j]; a[j] = t }
    for (i = 1; i <= NF; ++i) printf "%s%s", a[i], (i < NF ? " " : "\n") }'; }
rev() { out=; for x in "$@"; do out="$x $out"; done; echo $out; }
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "implementations ($nv): $(echo $V)"
    echo "sizes ($ns): $SIZES"
    echo "rounds $ROUNDS: $((ROUNDS*nv*ns)) program runs, pause $PAUSE s"
    for r in 1 2; do
        if [ $((r%2)) = 1 ]; then L=$SIZES; else L=$(rev $SIZES); fi
        j=0; for n in $L; do
            j=$((j+1)); [ $j -le 2 ] && echo "round $r, n = $n: $(shuffle $((1000*r+n)) $V)"
        done
    done
    exit 0
fi
. $S/guard.sh
mkdir -p res; : > $LOG.failures; : > $LOG.discards; : > $LOG.order; runs=0
echo "== start $(date '+%F %T'), $nv implementations, $ns sizes, rounds $ROUNDS, pause $PAUSE s, $(status)" | tee $LOG
for r in $(seq 1 $ROUNDS); do
    if [ $((r%2)) = 1 ]; then L=$SIZES; else L=$(rev $SIZES); fi
    for n in $L; do
        for v in $(shuffle $((1000*r+n)) $V); do
            if [ $v = accelerate ]; then p="env VECLIB_MAXIMUM_THREADS=1 ./bench2-$v"; else p=./bench2-$v; fi
            echo "$(date +%T) $r $n $v" >> $LOG.order
            measure res/$v-$n-$r.txt $p $SAMPLES 1 $n $n $n
            runs=$((runs+1)); nd=$(wc -l < $LOG.discards)
            if grep -q 'thermal state' $LOG.failures; then
                echo "ABORT $(date +%T): persistent throttling ($(tail -1 $LOG.failures))" | tee -a $LOG; exit 2
            fi
            if [ $runs -ge 20 ] && [ $((100*nd)) -gt $((${MAX_DISCARD_PERCENT:-10}*runs)) ]; then
                echo "ABORT $(date +%T): $nd of $runs runs discarded (thermal state not nominal)" | tee -a $LOG; exit 2
            fi
        done
    done
    echo "round $r done $(date '+%T'): $(status)" | tee -a $LOG
done
echo "discarded: $(wc -l < $LOG.discards), wrong results: $(cat res/*.txt | grep -c WRONG)" | tee -a $LOG
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi | tee -a $LOG
echo "== done $(date '+%F %T'), $(status)" | tee -a $LOG
