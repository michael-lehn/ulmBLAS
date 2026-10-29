#!/bin/sh
#  sample.sh BASE : statistical profiles (macOS sample, 1 ms interval, 5 s)
#  of DGEMM 2048 x 2048 x 2048 and 2048 x 64 x 2048: BLIS (cblas_dgemm of
#  prof-blis) and ulmBLAS (ulmBLAS::gemm of prof-master and prof-final),
#  under the thermal guard.  Output BASE/res/sample/NAME-SHAPE.txt (the
#  report of sample) and NAME-SHAPE-prof.txt (the output of prof).
set -u
BASE=${1:?}; S=$(cd "$(dirname "$0")" && pwd)
D=$BASE/bin; LOG=$BASE/res/sample; PAUSE=${PAUSE:-10}; . $S/guard.sh
mkdir -p $LOG
for shape in "2048 2048 2048" "2048 64 2048"; do
    t=$(echo $shape | tr ' ' x)
    for v in blis:cblas master:lib final:lib; do
        prog=prof-${v%%:*}; mode=${v##*:}
        cool
        $D/$prog -m $mode -s 25 -t 0.3 $shape > $LOG/${v%%:*}-$t-prof.txt &
        pid=$!
        sleep 1.5
        sample $pid 5 1 -mayDie -file $LOG/${v%%:*}-$t.txt > /dev/null 2>&1
        wait $pid
        echo "$v $t: $(cat $LOG/${v%%:*}-$t-prof.txt), thermal $(thermal_state)"
    done
done
