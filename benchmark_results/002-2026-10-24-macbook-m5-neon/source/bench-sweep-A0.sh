#!/bin/sh
#  bench-sweep.sh DIR : run A, block size sweep of the NEON micro kernel
#  (MC = 120, 240, 480, 960 x KC = 256, 512, 1024, 2048, NC = 4080), dgemm,
#  programs built by build-variants.sh in DIR/bench, results in
#  DIR/bench/res-A.  Single threaded (QoS user-interactive), ROUNDS rounds
#  with the order of the variants rotated, the FMA peak (peak) measured at
#  the start of every round.
#
#  [Archived as bench-sweep-A0.sh: the first version, without the thermal
#  guard.  Its results (run A0) are invalid because of thermal throttling,
#  see README.md; the results were moved from res-A to res-A0.]
set -u
D=${1:?}; B=$D/bench; ROUNDS=${ROUNDS:-3}
mkdir -p $B/res-A; cd $B || exit 1; : > failures-A
fail() { echo "FAILED (exit $1): $2" | tee -a failures-A; }
V=""; for mc in 120 240 480 960; do for kc in 256 512 1024 2048; do V="$V neon-$mc-$kc"; done; done
SWEEP="512 512 512 1024 1024 1024 2048 2048 2048 4096 4096 4096 2000 2000 256 2000 2000 32"
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
echo "== run A start $(date '+%F %T'), rounds $ROUNDS, $(uptime | sed 's/.*load/load/')"
for r in $(seq 1 $ROUNDS); do
    echo "round $r: $($D/peak 1), $(uptime | sed 's/.*load/load/')"
    for v in $(rot $((5*r-4)) $V); do
        ./bench-$v 7 1 $SWEEP > res-A/sweep-$v-$r.txt || fail $? "sweep $v $r"
    done
    echo "round $r done $(date '+%T'): $($D/peak 1), $(uptime | sed 's/.*load/load/')"
done
echo "wrong results: $(cat res-A/*.txt | grep -c WRONG)"
if [ -s failures-A ]; then cat failures-A; else echo "overall: all commands succeeded"; fi
