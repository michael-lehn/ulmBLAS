#!/bin/sh
#  bench-pairs.sh DIR : run A2, tightly interleaved comparison of candidate
#  block sizes of the NEON kernel (after the sweep of run A): CYCLES cycles,
#  in every cycle each candidate is measured once per size (short program
#  runs, order rotated from cycle to cycle), with the thermal guard
#  (guard.sh, PAUSE seconds before each run).  Slow changes of the machine
#  state (other load) then affect all candidates alike.  Results in
#  DIR/bench/res-A2/pair-VARIANT-CYCLE.txt.
set -u
D=${1:?}; B=$D/bench; CYCLES=${CYCLES:-8}; PAUSE=${PAUSE:-5}; LOG=$B/run-A2
S=$(cd "$(dirname "$0")" && pwd); . $S/guard.sh
mkdir -p $B/res-A2; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
$D/thermal 10 > $LOG.thermal & TH=$!; trap 'kill $TH' 0
V="neon-240-256 neon-120-1024 neon-240-1024 neon-480-1024 neon-960-1024"
echo "== run A2 start $(date '+%F %T'), cycles $CYCLES, pause $PAUSE s, $(status)"
for c in $(seq 1 $CYCLES); do
    for v in $(rot $c $V); do
        measure res-A2/pair-$v-$c.txt ./bench-$v 7 1 1024 1024 1024 2048 2048 2048 2000 2000 32
    done
    echo "cycle $c done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards), wrong results: $(cat res-A2/*.txt | grep -c WRONG)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
