#!/bin/sh
#  obhsw-heim.sh SRC W NAME=ARCHFLAGS ... : DGEMM of ulmBLAS with OpenBLAS's
#  HASWELL kernel against native OpenBLAS 0.3.34 (HASWELL), native BLIS 2.0
#  (haswell) and ulmBLAS's other AVX2/FMA kernels on heim (Intel Core
#  i5-12500).  Method of campaign 017 (kernels-heim-iter.sh):
#
#  - builds ulmBLAS (source tree SRC) once per NAME with the given ARCHFLAGS
#    (make cblas), links bench2.c and verify-large.c (directory of this
#    script) against each library and against the native OpenBLAS and BLIS
#    of $REFS (OBLIB, BLISLIB);
#  - checks that native OpenBLAS uses the HASWELL kernel (kernel-identity:
#    openblas_get_corename and the machine code of its dgemm_kernel against
#    the vendored kernel of ulmBLAS, data/logs/kernel-identity.txt);
#  - ROUNDS rounds (default 3), the order of the variants rotated per round;
#    every program run pinned to core CPU (default 3; its hyperthread
#    sibling must be idle as the rest of the machine), bench2 with 7
#    samples of at least 0.2 s after a warm-up of at least 0.3 s, median;
#    the first result of each size checked against a naive product
#    (sizes up to 2e8 multiply-adds), verify-large for the large sizes;
#  - load gate (gate): a program run starts only when no other process is
#    runnable and the 1-minute load average is below LOADMAX (default 1.8,
#    the previous run of this script contributes up to 1); load, temperature, frequency of CPU and
#    logged-in sessions are recorded before every run (logs/conditions.csv).
#
#      sh obhsw-heim.sh SRC W \
#          ulmblas-openblas-haswell_4x8="-mavx -mavx2 -mfma -DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8" ...
#
#  SIZES: main (default: the square, large and thin shapes of 017) or small.
#  EXTRA: programs of earlier experiments measured as further variants.
#
set -u
SRC=$1; W=$2; shift 2
D=$(cd $(dirname $0) && pwd)
REFS=${REFS:-/tmp/ulmblas-avx.tGdSUq/refs}
OBLIB=$REFS/openblas/lib/libopenblas.a; OBINC=$REFS/openblas/include
BLISLIB=$REFS/blis/lib/libblis.a; BLISINC=$REFS/blis/include/blis
CPU=${CPU:-3}; ROUNDS=${ROUNDS:-3}; LOADMAX=${LOADMAX:-1.8}; SIZES=${SIZES:-main}
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
    ar t v-$v/libcblas.a | grep -v '^[a-z0-9_]*\.o$' >> $L/compile-commands.txt
    ar t v-$v/libcblas.a | grep -q '^haswell_4x8\.o$' && echo "$v: libcblas.a contains haswell_4x8.o" >> $L/compile-commands.txt
    [ -f v-$v/interfaces/blas/C/build.cblas/haswell_4x8.o ] && cp v-$v/interfaces/blas/C/build.cblas/haswell_4x8.o $W/haswell_4x8-$v.o
    gcc -O2 -Iv-$v $D/bench2.c v-$v/libcblas.a -lstdc++ -lm -o bench-$v || fail $? "link $v"
    gcc -O2 -Iv-$v $D/verify-large.c v-$v/libcblas.a -lstdc++ -lm -o verify-$v || fail $? "link verify $v"
    rm -rf v-$v
    VARIANTS="$VARIANTS $v"
done
#  EXTRA="NAME=PATH ...": existing programs (bench-*, with the verify-*
#  program of the same directory if present), e.g. of an earlier experiment
for p in ${EXTRA:-}; do
    v=${p%%=*}; f=${p#*=}
    cp $f bench-$v || fail $? "copy $v"
    vf=$(dirname $f)/$(basename $f | sed 's/^bench-/verify-/')
    [ -x $vf ] && cp $vf verify-$v
    VARIANTS="$VARIANTS $v"
done
gcc -O2 -I$OBINC $D/bench2.c $OBLIB -lm -o bench-openblas || fail $? "link openblas"
gcc -O2 -I$OBINC $D/verify-large.c $OBLIB -lm -o verify-openblas || fail $? "link verify openblas"
gcc -O2 -I$BLISINC $D/bench2.c $BLISLIB -lpthread -lm -o bench-blis || fail $? "link blis"
gcc -O2 -I$BLISINC $D/verify-large.c $BLISLIB -lpthread -lm -o verify-blis || fail $? "link verify blis"
VARIANTS="$VARIANTS openblas blis"
sha256sum bench-* verify-* $OBLIB $BLISLIB > $L/programs.sha256

#  native OpenBLAS: core selected and machine code of its DGEMM kernel
{
    cat > corename.c <<'EOF'
#include <stdio.h>
char *openblas_get_corename(void);
char *openblas_get_config(void);
int main(void) { printf("corename %s\nconfig %s\n", openblas_get_corename(), openblas_get_config()); return 0; }
EOF
    gcc corename.c $OBLIB -lm -o corename && ./corename
    grep -hE '^(CORE|LIBCORE|DYNAMIC_ARCH|USE_THREAD|TARGET)=' $REFS/src/OpenBLAS-0.3.34/Makefile.conf* | sort -u
    mkdir -p ob-objs && (cd ob-objs && ar x $OBLIB dgemm_kernel.o)
    for o in ob-objs/dgemm_kernel.o $W/haswell_4x8-*.o; do
        [ -f "$o" ] || continue
        objcopy -O binary --only-section=.text $o $o.text
        echo "$(sha256sum < $o.text | cut -c1-64) $(wc -c < $o.text) bytes .text of $(basename $o)"
    done
    for o in $W/haswell_4x8-*.o; do
        [ -f "$o" ] && { cmp -s ob-objs/dgemm_kernel.o.text $o.text \
            && echo "IDENTICAL: .text of native dgemm_kernel.o and $(basename $o)" \
            || echo "DIFFERENT: .text of native dgemm_kernel.o and $(basename $o)"; }
    done
} > $L/kernel-identity.txt 2>&1

{
  echo "date $(date -Iseconds)"; uname -a; cat /etc/debian_version
  gcc --version | head -1; g++ --version | head -1; as --version | head -1
  grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo
  echo "cpu $CPU, siblings $(cat /sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list)"
  echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor) no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  echo "load $(cat /proc/loadavg)"; who | wc -l
  echo "variants:$VARIANTS"; echo "sizes $SIZES, rounds $ROUNDS, loadmax $LOADMAX"
} > $L/system.txt 2>&1

if [ "$SIZES" = small ]; then
    SQ=""; for n in 2 3 4 5 6 7 8 10 12 16 20 24 31 32 40 48 50 64 80 96 100 128 150 160 192 200 256 300 384 400 512; do SQ="$SQ $n $n $n"; done
    LARGE=""; THIN=""
else
    SQ="7 7 7 31 31 31 64 64 64 100 100 100 128 128 128 199 199 199 256 256 256 384 384 384 512 512 512 768 768 768"
    LARGE="1000 1000 1000 1024 1024 1024 1500 1500 1500 2000 2000 2000 2048 2048 2048 3000 3000 3000 4000 4000 4000 4096 4096 4096"
    THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
fi
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
#  gate: no other runnable process (procs_running of /proc/stat, which
#  counts the reading process itself, at most 1 in 3 samples 1 s apart) and
#  1-minute load average below LOADMAX (the previous program run of this
#  script contributes up to 1)
idle() { [ "$(awk '$1=="procs_running" {print $2}' /proc/stat)" -le 1 ]; }
gate() {
    until awk -v l=$(cut -d' ' -f1 /proc/loadavg) -v x=$LOADMAX 'BEGIN { exit !(l < x) }' \
          && idle && sleep 1 && idle && sleep 1 && idle; do
        sleep 10
    done
}
echo "time,kind,variant,round,load1,package_temp_C,cpu_MHz,sessions" > $L/conditions.csv
echo "start $(date -Iseconds)" >> $L/run.log
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        gate; cond main $v $r
        taskset -c $CPU ./bench-$v 7 1 $SQ $LARGE $THIN > res/$SIZES-$v-$r.txt || fail $? "$SIZES $v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
if [ "$SIZES" != small ]; then
    for v in $VARIANTS; do
        gate; taskset -c $CPU ./verify-$v 1000 1500 2000 3000 4000 4096 > res/verify-$v.txt 2>&1 || fail $? "verify $v"
    done
fi
echo "wrong results: $(cat res/$SIZES-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
