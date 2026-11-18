#!/bin/sh
#  compare.sh "OPS" "SIZES" LIB...: one line per library and operation,
#  GFLOP/s for the sizes on 1 and 6 threads (THREADS), median of TRIALS
#  layouts (see l2bench.c)
ops=$1
sizes=$2
shift 2
for t in ${THREADS:-1 6}; do
    for op in $ops; do
	for lib in "$@"; do
	    printf '%-28s %d %-9s' "$(basename $lib)" $t $op
	    OPENBLAS_NUM_THREADS=$t ULMBLAS_NUM_THREADS=$t \
		./l2bench $lib $op $sizes | awk '{printf " %7.1f", $3}'
	    echo
	done
    done
done
