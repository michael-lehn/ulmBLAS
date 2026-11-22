#!/bin/sh
#
#  evaluate.sh OUT: the result of run.sh in OUT (summary.txt, list.all).
#  Exit status 0 if every run of the list finished with status 0, at least
#  one PASS and no FAIL, ERROR (stderr), invalid option ("ATLAS usage") or
#  NoMEM; otherwise 1, with the runs that did not.
#
set -u
O=$1
S=$O/summary.txt
[ -f "$S" ] || { echo "error: $S not found" >&2; exit 2; }
runs=$(grep -c . "$O/list.all")
awk -v runs="$runs" '
    $1 == "done" { done = 1; next }
    {
        ++n; pass += $5; fail += $7
        if ($3 != 0 || $5 == 0 || $7 > 0 || $9 > 0 || $11 > 0 || $13 > 0) {
            ++bad
            print "FAIL  " $0
        }
    }
    END {
        printf "check-large: %d of %d runs, PASS %d, FAIL %d, runs with problems %d\n",
               n, runs, pass, fail, bad
        if (!done || n != runs) {
            print "check-large: incomplete (" n " of " runs " runs)"
            exit 1
        }
        exit bad > 0
    }' "$S"
