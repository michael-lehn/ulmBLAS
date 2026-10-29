#!/bin/sh
#  kernels-heim.sh SRC WORKSPACE: DGEMM with the BLIS-derived micro kernels
#  that the Intel Core i5-12500 of heim can execute (AVX2/FMA, AVX, SSE3),
#  single threaded on core 3, with the methodology of the heim campaigns
#  (bench2.c: 7 samples of at least 0.2 s after a warm-up of at least 0.3 s,
#  median; first result checked; rounds with rotated order).
#
#  SRC is the ulmBLAS source tree (with ulmblas/level3/ukernel/blis/),
#  WORKSPACE a new directory for builds and results; bench2.c and
#  verify-large.c are taken from the directory of this script.  Variants:
#
#    ulmblas-avx              ulmBLAS's AVX/FMA kernel (6 x 8, a port of
#                             BLIS haswell), source SRC without a BLIS kernel
#    ulmblas-sse              ulmBLAS's own SSE kernel (4 x 4), source SRC
#    ulmblas-haswell_6x8      ulmBLAS (source SRC) with the vendored BLIS
#    ulmblas-haswell_8x6      kernel NAME and its default block sizes (those
#    ulmblas-sandybridge_8x4  of the BLIS configuration that uses it)
#    ulmblas-penryn_4x4
#    ulmblas-avx-febb852      ulmBLAS febb852 (program of 2026-10-26,
#                             unchanged)
#    blis-haswell             native BLIS 2.0, configuration haswell
#                             (program of 2026-10-24, unchanged)
#    blis-sandybridge         native BLIS 2.0, configurations sandybridge and
#    blis-penryn              penryn (built here from blis-2.0.tar.gz of
#                             2026-10-24, configured as there)
#    openblas                 OpenBLAS 0.3.34 (program of 2026-10-24,
#                             unchanged)
#
set -u
SRC=$1; W=$2
D=$(cd $(dirname $0) && pwd)              # bench2.c, verify-large.c
mkdir -p $W && cd $W || exit 1
OLD1=/tmp/ulmblas-avx.tGdSUq/bench        # programs of 2026-10-24
OLD2=/tmp/ulmblas-intel2.2yi6R9/bench     # programs of 2026-10-26
L=$W/logs; mkdir -p $L res
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
CPU=3; ROUNDS=${ROUNDS:-3}

flags() { case $1 in
    ulmblas-avx)             echo "-mavx -mfma";;
    ulmblas-sse)             echo "-msse3 -mfpmath=sse";;
    ulmblas-haswell_6x8)     echo "-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_6x8";;
    ulmblas-haswell_8x6)     echo "-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_8x6";;
    ulmblas-sandybridge_8x4) echo "-mavx -DULMBLAS_DGEMM_UKERNEL_blis_sandybridge_8x4";;
    ulmblas-penryn_4x4)      echo "-msse3 -mfpmath=sse -DULMBLAS_DGEMM_UKERNEL_blis_penryn_4x4";;
esac; }
BUILT="ulmblas-avx ulmblas-sse ulmblas-haswell_6x8 ulmblas-haswell_8x6 ulmblas-sandybridge_8x4 ulmblas-penryn_4x4"
VARIANTS="$BUILT ulmblas-avx-febb852 blis-haswell blis-sandybridge blis-penryn openblas"

: > $L/compile-commands.txt
for v in $BUILT; do
    rm -rf v-$v && mkdir v-$v
    (cd $SRC && find . -path ./test/build -prune -o -type f -print | grep -vE '^\./(benchmark_results|investigations|test/build)/' | tar cf - -T -) | tar xf - -C v-$v
    (cd v-$v && make -j10 cblas ARCHFLAGS="`flags $v`") > $L/build-$v.log 2>&1 || fail $? "build $v"
    echo "$v: $(cat v-$v/interfaces/blas/C/build.cblas/compile.cmd)" >> $L/compile-commands.txt
    gcc -O2 -Iv-$v $D/bench2.c v-$v/libcblas.a -lstdc++ -lm -o bench-$v || fail $? "link $v"
    gcc -O2 -Iv-$v $D/verify-large.c v-$v/libcblas.a -lstdc++ -lm -o verify-$v || fail $? "link verify $v"
done
#  native BLIS sandybridge and penryn, as BLIS haswell on 2026-10-24
for c in sandybridge penryn; do
    rm -rf blis-src-$c && mkdir blis-src-$c
    tar xzf /tmp/ulmblas-avx.tGdSUq/refs/src/blis-2.0.tar.gz -C blis-src-$c --strip-components=1
    (cd blis-src-$c && ./configure --prefix=$W/refs/blis-$c --disable-shared --enable-cblas -t no $c \
        && make -j10 && make install) > $L/build-blis-$c.log 2>&1 || fail $? "build blis-$c"
    gcc -O2 -I$W/refs/blis-$c/include/blis $D/bench2.c $W/refs/blis-$c/lib/libblis.a -lm -lpthread -o bench-blis-$c || fail $? "link blis-$c"
    gcc -O2 -I$W/refs/blis-$c/include/blis $D/verify-large.c $W/refs/blis-$c/lib/libblis.a -lm -lpthread -o verify-blis-$c || fail $? "link verify blis-$c"
    rm -rf blis-src-$c
done
cp $OLD2/bench-avx-cur bench-ulmblas-avx-febb852 && cp $OLD2/verify-avx-cur verify-ulmblas-avx-febb852 || fail $? "copy avx-febb852"
cp $OLD1/bench-blis bench-blis-haswell && cp $OLD1/bench-openblas bench-openblas || fail $? "copy native"
cp $OLD2/verify-blis verify-blis-haswell && cp $OLD2/verify-openblas verify-openblas || fail $? "copy verify native"
sha256sum bench-* verify-* > $L/programs.sha256

{
  echo "date $(date -Iseconds)"; uname -a; cat /etc/debian_version
  gcc --version | head -1; grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo
  echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor) no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  echo "load $(cat /proc/loadavg)"; who | wc -l
} > $L/system.txt 2>&1

SQ="7 7 7 31 31 31 64 64 64 100 100 100 128 128 128 199 199 199 256 256 256 384 384 384 512 512 512 768 768 768"
LARGE="1000 1000 1000 1024 1024 1024 1500 1500 1500 2000 2000 2000 2048 2048 2048 3000 3000 3000 4000 4000 4000 4096 4096 4096"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
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
echo "time,kind,variant,round,load1,package_temp_C,cpu3_MHz,sessions" > $L/conditions.csv

echo "start $(date -Iseconds)" >> $L/run.log
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        cond main $v $r; taskset -c $CPU ./bench-$v 7 1 $SQ $LARGE $THIN > res/main-$v-$r.txt || fail $? "main $v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
for v in $VARIANTS; do
    taskset -c $CPU ./verify-$v 1000 1500 2000 3000 4000 4096 > res/verify-$v.txt 2>&1 || fail $? "verify $v"
done
echo "wrong results: $(cat res/main-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
