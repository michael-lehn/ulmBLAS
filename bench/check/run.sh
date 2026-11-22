#!/bin/sh
#  run.sh BIN OUT THREADS SLOTS [PATTERN]: the runs of list.sh (those whose
#  name matches the grep pattern PATTERN, default all), SLOTS testers at a
#  time, each with ULMBLAS_NUM_THREADS=THREADS (all: the default of
#  ulmBLAS, ULMBLAS_NUM_THREADS and OMP_NUM_THREADS removed).  With taskset
#  (Linux) and THREADS a number, slot k runs on the CPUs k*THREADS, ...,
#  (k+1)*THREADS-1.  OUT/summary.txt: per run status, PASS, FAIL, ERROR
#  (stderr lines), usage (invalid options), NoMEM, seconds; "done" at the end.
set -u
D=$(cd "$(dirname "$0")" && pwd)
BIN=$(cd "$1" && pwd)
mkdir -p "$2"
O=$(cd "$2" && pwd)
TH=$3
SLOTS=$4
PAT=${5:-.}
#  the long runs (banded level 2, small sizes) first
sh "$D/list.sh" | grep -e "$PAT" > "$O/list.all"
{ grep -E -e '-l2-.b[ms]v-small ' "$O/list.all"; grep -v -E -e '-l2-.b[ms]v-small ' "$O/list.all"; } > "$O/queue"
: > "$O/summary.txt"
: > "$O/taken"
if [ "$TH" = all ]; then
    unset ULMBLAS_NUM_THREADS OMP_NUM_THREADS
else
    export ULMBLAS_NUM_THREADS=$TH
fi
TS=""
command -v taskset > /dev/null 2>&1 && [ "$TH" != all ] && TS=taskset

take()
{
    #  the next line of the queue (a directory as the lock)
    while ! mkdir "$O/lock" 2> /dev/null; do sleep 0.05; done
    n=$(($(wc -l < "$O/taken") + 1))
    line=$(sed -n "${n}p" "$O/queue")
    [ -n "$line" ] && echo x >> "$O/taken"
    rmdir "$O/lock"
    printf '%s' "$line"
}

worker()
{
    k=$1
    while line=$(take) && [ -n "$line" ]; do
        set -- $line
        name=$1; tester=$2; shift 2
        t0=$(date +%s)
        if [ -n "$TS" ]; then
            first=$((k * TH)); last=$((first + TH - 1))
            taskset -c $first-$last "$BIN/$tester" "$@" > "$O/$name.out" 2> "$O/$name.err"
        else
            "$BIN/$tester" "$@" > "$O/$name.out" 2> "$O/$name.err"
        fi
        st=$?
        t1=$(date +%s)
        echo "$name status $st PASS $(grep -c ' PASS *$' "$O/$name.out")" \
             "FAIL $(grep -c ' FAIL *$' "$O/$name.out")" \
             "ERROR $(grep -c 'ERROR' "$O/$name.err")" \
             "usage $(cat "$O/$name.out" "$O/$name.err" | grep -c 'ATLAS usage')" \
             "NoMEM $(grep -c 'NoMEM' "$O/$name.out")" \
             "sec $((t1 - t0))" >> "$O/summary.txt"
    done
}

k=0
while [ $k -lt "$SLOTS" ]; do
    worker $k &
    k=$((k + 1))
done
wait
echo done >> "$O/summary.txt"
