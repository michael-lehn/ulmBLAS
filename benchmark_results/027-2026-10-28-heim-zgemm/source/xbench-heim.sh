#!/bin/sh
#  xbench-heim.sh PREC SRC W NAME=ARCHFLAGS ... : single-threaded ?GEMM
#  (PREC = s, d, c, z) of ulmBLAS builds against native OpenBLAS 0.3.34
#  (HASWELL) and BLIS 2.0 (haswell) on heim (Intel Core i5-12500).  Method
#  of campaign 024 (obhsw-heim.sh):
#
#  - builds ulmBLAS (source tree SRC) once per NAME with the given ARCHFLAGS
#    (make cblas), links benchx.c (directory of this script, -DPREC=PREC)
#    against each library and against native OpenBLAS and BLIS of $REFS;
#  - records which kernel native OpenBLAS and BLIS use for PREC
#    (logs/kernel-identity.txt, kernel-info.c) and the selection macros and
#    kernel symbols of each ulmBLAS library (logs/compile-commands.txt);
#  - ROUNDS rounds (default 3), the order of the variants rotated per round;
#    variants listed in ONCE (e.g. the reference kernel, much slower) are
#    measured in the first round only; every program run pinned to core
#    CPU (default 3; its hyperthread sibling must be idle as the rest of
#    the machine); benchx with SAMPLES (default 3) samples of at least
#    0.2 s after a warm-up of at least 0.3 s, median; every result checked
#    (400 elements of C against sums in long double);
#  - load gate (gate): a program run starts only when no other process is
#    runnable and the 1-minute load average is below LOADMAX (default 1.8);
#    load, temperature, frequency of CPU and logged-in sessions are recorded
#    before every run (logs/conditions.csv).
#
#  Sizes: m = n = k = 50, 100, ..., 4000 (SIZES overrides).
#
set -u
PREC=$1; SRC=$2; W=$3; shift 3
D=$(cd $(dirname $0) && pwd)
REFS=${REFS:-/tmp/ulmblas-avx.tGdSUq/refs}
OBLIB=$REFS/openblas/lib/libopenblas.a; OBINC=$REFS/openblas/include
BLISLIB=$REFS/blis/lib/libblis.a; BLISINC=$REFS/blis/include/blis
CPU=${CPU:-3}; ROUNDS=${ROUNDS:-3}; LOADMAX=${LOADMAX:-1.8}; SAMPLES=${SAMPLES:-3}
ONCE=${ONCE:-}
SIZES=${SIZES:-$(seq 50 50 4000)}
mkdir -p $W && cd $W || exit 1
L=$W/logs; mkdir -p $L res
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
VARIANTS=""
: > $L/compile-commands.txt
while [ $# -gt 0 ]; do
    v=${1%%=*}; f=${1#*=}; shift
    rm -rf v-$v && mkdir v-$v
    (cd $SRC && find . -path ./test/build -prune -o -type f -print | grep -vE '^\./(benchmark_results|investigations|test/build)/' | tar cf - -T -) | tar xf - -C v-$v
    (cd v-$v && make -j10 cblas ARCHFLAGS="$f") > $L/build-$v.log 2>&1 || fail $? "build $v"
    echo "$v: $(cat v-$v/interfaces/blas/C/build.cblas/compile.cmd)" >> $L/compile-commands.txt
    #  kernel functions in the library (vendored kernels are inline; the
    #  OpenBLAS kernels are noinline and appear as symbols)
    nm -C v-$v/libcblas.a 2>/dev/null | grep -E ' [TtWw] .*(bli_[sdcz]gemm|[sdcz]gemm_kernel)' | sort -u | sed "s/^/$v: symbol /" >> $L/compile-commands.txt
    gcc -O2 -DPREC=$PREC -Iv-$v $D/benchx.c v-$v/libcblas.a -lstdc++ -lm -o bench-$v || fail $? "link $v"
    rm -rf v-$v
    VARIANTS="$VARIANTS $v"
done
gcc -O2 -DPREC=$PREC -I$OBINC $D/benchx.c $OBLIB -lm -o bench-openblas || fail $? "link openblas"
gcc -O2 -DPREC=$PREC -I$BLISINC $D/benchx.c $BLISLIB -lpthread -lm -o bench-blis || fail $? "link blis"
VARIANTS="$VARIANTS openblas blis"
sha256sum bench-* $OBLIB $BLISLIB > $L/programs.sha256

#  kernels of native OpenBLAS and BLIS for PREC
{
    gcc -O2 -DPREC=$PREC -DOPENBLAS $D/kernel-info.c $OBLIB -lm -o info-openblas && ./info-openblas
    grep -hE '^(CORE|LIBCORE|DYNAMIC_ARCH|USE_THREAD|TARGET)=' $REFS/src/OpenBLAS-0.3.34/Makefile.conf* | sort -u
    ar t $OBLIB | grep -E "^${PREC}gemm_(kernel|small|beta|incopy|itcopy|oncopy|otcopy)" | sort | tr '\n' ' '; echo
    gcc -O2 -DPREC=$PREC -DBLIS -I$BLISINC $D/kernel-info.c $BLISLIB -lpthread -lm -o info-blis && ./info-blis
} > $L/kernel-identity.txt 2>&1

{
  echo "date $(date -Iseconds)"; uname -a; cat /etc/debian_version
  gcc --version | head -1; g++ --version | head -1; as --version | head -1
  grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo
  echo "cpu $CPU, siblings $(cat /sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list)"
  echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor) no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  echo "load $(cat /proc/loadavg)"; echo "sessions $(who | wc -l)"
  echo "prec $PREC; variants:$VARIANTS; once: $ONCE"
  echo "rounds $ROUNDS, samples $SAMPLES, loadmax $LOADMAX"
  echo "sizes $(echo $SIZES | tr '\n' ' ')"
} > $L/system.txt 2>&1

rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
HWMON=
for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = coretemp ] && HWMON=$h
done
cond() {
    t=; [ -n "$HWMON" ] && t=$(($(cat $HWMON/temp1_input)/1000))
    f=$(awk -v c=$CPU '$1=="processor"{p=$3} $1=="cpu" && $2=="MHz" && p==c {print $4}' /proc/cpuinfo)
    echo "$(date +%s),$1,$2,$3,$(cut -d' ' -f1 /proc/loadavg),$t,$f,$(who | wc -l)" >> $L/conditions.csv
}
idle() { [ "$(awk '$1=="procs_running" {print $2}' /proc/stat)" -le 1 ]; }
gate() {
    until awk -v l=$(cut -d' ' -f1 /proc/loadavg) -v x=$LOADMAX 'BEGIN { exit !(l < x) }' \
          && idle && sleep 1 && idle && sleep 1 && idle; do
        sleep 10
    done
}
once() { case " $ONCE " in *" $1 "*) return 0;; esac; return 1; }
echo "time,kind,variant,round,load1,package_temp_C,cpu_MHz,sessions" > $L/conditions.csv
echo "start $(date -Iseconds)" >> $L/run.log
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        if [ $r -gt 1 ] && once $v; then continue; fi
        gate; cond main $v $r
        taskset -c $CPU ./bench-$v $SAMPLES $SIZES > res/$PREC-$v-$r.txt || fail $? "$v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
echo "wrong results: $(cat res/$PREC-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
