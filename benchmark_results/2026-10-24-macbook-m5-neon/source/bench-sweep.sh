#!/bin/sh
#  bench-sweep.sh DIR : run A, block size sweep of the NEON micro kernel
#  (MC = 120, 240, 480, 960 x KC = 256, 512, 1024, NC = 4080), dgemm, with
#  the thermal guard (guard.sh).  Programs built by build-variants.sh in
#  DIR/bench, results in DIR/bench/res-A (one file per kind of sizes,
#  variant and round: sweep-VARIANT-ROUND.txt holds 512, 1024, 2048,
#  sweep4096-*, sweepthin-* the other sizes).  Single threaded (QoS
#  user-interactive), ROUNDS rounds with the order of the variants rotated.
set -u
D=${1:?}; B=$D/bench; ROUNDS=${ROUNDS:-3}; LOG=$B/run-A
S=$(cd "$(dirname "$0")" && pwd); . $S/guard.sh
mkdir -p $B/res-A; cd $B || exit 1; : > $LOG.failures; : > $LOG.discards
$D/thermal 10 > $LOG.thermal & TH=$!; trap 'kill $TH' 0
V=""; for mc in 120 240 480 960; do for kc in 256 512 1024; do V="$V neon-$mc-$kc"; done; done
echo "== run A start $(date '+%F %T'), rounds $ROUNDS, pause $PAUSE s, $(status)"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $((4*r-3)) $V); do
        measure res-A/sweep-$v-$r.txt     ./bench-$v 7 1 512 512 512 1024 1024 1024 2048 2048 2048
        measure res-A/sweep4096-$v-$r.txt ./bench-$v 7 1 4096 4096 4096
        measure res-A/sweepthin-$v-$r.txt ./bench-$v 7 1 2000 2000 256 2000 2000 32
    done
    echo "round $r done $(date '+%T'): $(status)"
done
echo "discarded: $(wc -l < $LOG.discards), wrong results: $(cat res-A/*.txt | grep -c WRONG)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
