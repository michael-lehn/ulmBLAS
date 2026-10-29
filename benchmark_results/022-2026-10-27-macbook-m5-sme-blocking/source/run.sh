#!/bin/sh
#  run.sh B E : DGEMM (column major, NoTrans, NoTrans, alpha = 1, beta = 1,
#  one thread) of the configurations VARIANTS (programs B/bench2-V) for
#  the shapes SHAPES (m_n_k ...); results in B/E/res/V-m_n_k-R.txt, logs
#  B/E/run*.  The methodology of the campaigns 2026-10-25-macbook-m5-sme-
#  square and 2026-10-26-macbook-m5-sme-small (compare.sh):
#
#  Interleaved per shape: in every round and for every shape all
#  configurations run one after the other, each in its own program run
#  (bench2 SAMPLES 1 m n k: warm-up >= 0.3 s, SAMPLES samples of >= 0.2 s,
#  median, min, max), in a random order per round and shape (seed 1000 *
#  round + index of the shape); the order of the shapes alternates between
#  rounds.  Every program run under the thermal guard (guard.sh: pause
#  PAUSE s, thermal state nominal before and after, otherwise repeated).
#  The SHA-256 of every program is logged before the first run.
#
#  Environment: VARIANTS (required), SHAPES (required), ROUNDS (default
#  3), PAUSE (default 3), SAMPLES (default 5), DRY_RUN=1 (print the number
#  of program runs only), ROUND0 (default 0: the rounds are numbered
#  ROUND0+1 ... ROUND0+ROUNDS, for further rounds in a new directory E).
set -u
B=${1:?usage: run.sh B E}; E=${2:?usage: run.sh B E}
ROUNDS=${ROUNDS:-3}; ROUND0=${ROUND0:-0}; SAMPLES=${SAMPLES:-5}; PAUSE=${PAUSE:-3}
V=${VARIANTS:?}; SHAPES=${SHAPES:?}
S=$(cd "$(dirname "$0")" && pwd); D=$B
cd $B || exit 1
nv=$(echo $V | wc -w); ns=$(echo $SHAPES | wc -w)
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$E: $nv configurations x $ns shapes x $ROUNDS rounds = $((nv*ns*ROUNDS)) program runs"
    exit 0
fi
for v in $V; do [ -x bench2-$v ] || { echo "missing bench2-$v"; exit 1; }; done
mkdir -p $E/res || exit 1; LOG=$B/$E/run
. $S/guard.sh
shuffle() { seed=$1; shift; echo "$@" | awk -v s=$seed '{ srand(s); for (i = 1; i <= NF; ++i) a[i] = $i;
    for (i = NF; i > 1; --i) { j = int(rand() * i) + 1; t = a[i]; a[i] = a[j]; a[j] = t }
    for (i = 1; i <= NF; ++i) printf "%s%s", a[i], (i < NF ? " " : "\n") }'; }
rev() { out=; for x in "$@"; do out="$x $out"; done; echo $out; }
: > $LOG.failures; : > $LOG.discards; : > $LOG.order
for v in $V; do shasum -a 256 bench2-$v; done > $LOG.programs
echo "== start $(date '+%F %T'), $nv configurations, $ns shapes, rounds $ROUNDS, pause $PAUSE s, $(status)" | tee $LOG
for r in $(seq $((ROUND0+1)) $((ROUND0+ROUNDS))); do
    if [ $((r%2)) = 1 ]; then L=$SHAPES; else L=$(rev $SHAPES); fi
    i=0
    for s in $L; do
        i=$((i+1))
        set -- $(echo $s | tr _ ' ')
        for v in $(shuffle $((1000*r+i)) $V); do
            echo "$(date +%T) $r $s $v" >> $LOG.order
            measure $E/res/$v-$s-$r.txt ./bench2-$v $SAMPLES 1 $1 $2 $3
            if grep -q 'thermal state' $LOG.failures; then
                echo "ABORT $(date +%T): persistent throttling" | tee -a $LOG; exit 2
            fi
        done
    done
    echo "round $r done $(date '+%T'): $(status)" | tee -a $LOG
done
echo "discarded: $(wc -l < $LOG.discards), wrong results: $(cat $E/res/*.txt | grep -c WRONG)" | tee -a $LOG
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi | tee -a $LOG
echo "== done $(date '+%F %T'), $(status)" | tee -a $LOG
