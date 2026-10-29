#  guard.sh: measurement harness of the macOS benchmarks (sourced by the run
#  scripts).  The MacBook Air has passive cooling: after a few minutes of
#  sustained load on one core macOS raises the thermal state from nominal
#  to fair and the clock frequency drops (DGEMM about 15% slower, see
#  README.md).  Therefore every measurement is a short program run that
#
#  - starts only when the thermal state is nominal, after a pause of PAUSE
#    seconds (default 10),
#  - is accepted only if the thermal state is nominal before and after the
#    run; otherwise it is discarded (logged in DISCARDS) and repeated, at
#    most TRIES times (default 3).
#
#  Needs D (directory of the programs peak and thermal) and LOG (log file).
PAUSE=${PAUSE:-10}; TRIES=${TRIES:-3}
thermal_state() { $D/thermal | awk '{print $3}'; }
cool() {
    until [ "$(thermal_state)" = nominal ]; do sleep 5; done
    sleep $PAUSE
}
#  measure OUT CMD...: run CMD (standard output to OUT) under the guard
measure() {
    out=$1; shift; try=1
    while :; do
        cool
        t0=$(thermal_state)
        "$@" > $out.tmp; rc=$?
        t1=$(thermal_state)
        if [ $rc -ne 0 ]; then
            echo "FAILED (exit $rc): $out" | tee -a $LOG.failures; rm -f $out.tmp; return 1
        fi
        if [ $t0 = nominal ] && [ $t1 = nominal ]; then
            mv $out.tmp $out; return 0
        fi
        echo "$(date +%T) discarded $out (try $try, thermal $t0 -> $t1)" >> $LOG.discards
        if [ $try -ge $TRIES ]; then
            echo "FAILED (thermal state, $TRIES tries): $out" | tee -a $LOG.failures
            mv $out.tmp $out.thermal; return 1
        fi
        try=$((try+1))
    done
}
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
status() { echo "$($D/peak 1), thermal $(thermal_state), $(uptime | sed 's/.*load/load/')"; }
