#!/bin/sh
#  run-jobs.sh BASE RUN JOBS ROUNDS : run the jobs of the job list JOBS (from
#  a jobs-*.sh script) ROUNDS times under the thermal guard (guard.sh), the
#  order rotated from round to round.  Programs in BASE/bin, output
#  BASE/res/RUN/rROUND/NAME.txt, logs BASE/res/RUN.*.
set -u
BASE=${1:?}; RUN=${2:?}; JOBS=${3:?}; ROUNDS=${4:-3}
S=$(cd "$(dirname "$0")" && pwd)
D=$BASE/bin; LOG=$BASE/res/$RUN; PAUSE=${PAUSE:-5}; . $S/guard.sh
O=$BASE/res/$RUN
mkdir -p $O; : > $LOG.failures; : > $LOG.discards
sh $S/$JOBS > $O/jobs.txt
N=$(wc -l < $O/jobs.txt)
$D/thermal 10 > $LOG.thermal.txt 2>&1 & TP=$!
echo "== $RUN start $(date '+%F %T'), $N jobs x $ROUNDS rounds, $(status)"
r=1
while [ $r -le $ROUNDS ]; do
    mkdir -p $O/r$r
    off=$(( ( (r-1) * N / ROUNDS ) + 1 ))
    { tail -n +$off $O/jobs.txt; [ $off -gt 1 ] && head -n $((off-1)) $O/jobs.txt; } > $O/order-r$r.txt
    while read name prog args; do
        measure $O/r$r/$name.txt $D/$prog $args
    done < $O/order-r$r.txt
    echo "== round $r done $(date '+%T'), $(status)"
    r=$((r+1))
done
kill $TP; wait $TP 2>/dev/null
echo "== $RUN done $(date '+%T'), $(status)"
if [ -s $LOG.failures ]; then cat $LOG.failures; else echo "overall: all commands succeeded"; fi
