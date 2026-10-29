#!/bin/sh
#  zbench-mac.sh SRC W: single-threaded ZGEMM on the MacBook Air (Apple M5),
#  m = n = k = 50, 100, ..., 4000 (SIZES overrides), 8mnk flops.
#
#  Variants: ulmBLAS (source tree SRC) with its NEON kernels (default
#  build) and with its SME2 kernels (-mcpu=apple-m4); Apple Accelerate
#  (VECLIB_MAXIMUM_THREADS=1, the reference for the ratios), OpenBLAS
#  (Homebrew, OPENBLAS_NUM_THREADS=1) and Eigen (pkg-config eigen3, its own
#  product C.noalias() += A*B, no OpenMP).  Programs: bench/gemm/benchx.c
#  and bench-eigen.cpp of SRC (performance core QoS, warm-up >= 0.3 s,
#  SAMPLES samples >= 0.2 s, median, 400 elements of C checked).
#
#  Method of campaigns 007 and 022: every measurement is one program run
#  for one size under the thermal guard (guard.sh: thermal state nominal
#  before and after the run, otherwise discarded and repeated; a run starts
#  only when the 1-minute load average is below LOADMAX, default 4, and
#  after a pause of PAUSE seconds).  For each round and size the variants
#  run in a random order (seeded with round and size).  The FMA throughput
#  of the core (peak) and the load are logged before every size.
#
#  Results: W/res/VARIANT-ROUND-N.txt, logs in W/logs.
#
set -u
SRC=$1; W=$2
S=$(cd $(dirname $0) && pwd)
ROUNDS=${ROUNDS:-3}; SAMPLES=${SAMPLES:-3}; ONCE=${ONCE:-}
SIZES=${SIZES:-$(awk 'BEGIN { for (n=50; n<=4000; n+=50) print n }')}
export LOADMAX=${LOADMAX:-4} PAUSE=${PAUSE:-10}
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p $W/bin $W/res $W/logs && cd $W || exit 1
LOG=$W/logs/run; : > $LOG.failures
D=$W/bin                                # guard.sh: thermal, peak
. $S/guard.sh
fail() { echo "FAILED (exit $1): $2" | tee -a $LOG.failures; }

clang -O2 -framework Foundation $S/thermal.m -o bin/thermal || fail $? thermal
cc -O2 $S/peak.c -o bin/peak || fail $? peak

#  ulmBLAS builds
VARIANTS=
for spec in "ulmblas-neon|" "ulmblas-sme|-mcpu=apple-m4"; do
    v=${spec%%|*}; f=${spec#*|}
    rm -rf v-$v && mkdir v-$v
    (cd $SRC && find . -path ./test/build -prune -o -type f -print \
        | grep -vE '^\./(benchmark_results|investigations|test/build|bench/gemm/results|\.git)/' \
        | tar cf - -T -) | tar xf - -C v-$v
    (cd v-$v && make -j8 cblas ARCHFLAGS="$f") > logs/build-$v.log 2>&1 || fail $? "build $v"
    echo "$v: $(cat v-$v/interfaces/blas/C/build.cblas/compile.cmd)" >> logs/compile-commands.txt
    cc -O2 -DPREC=z -DBENCH_LIB_ulmblas -Iv-$v -c v-$v/bench/gemm/benchx.c -o bin/$v.o \
        && c++ bin/$v.o v-$v/libcblas.a -o bin/bench-$v || fail $? "link $v"
    c++ -std=c++11 $f -Iv-$v v-$v/bench/gemm/ulmblas-kernels.cc -o bin/kernels-$v \
        && bin/kernels-$v > logs/kernels-$v.txt || fail $? "kernels $v"
    VARIANTS="$VARIANTS $v"
done
#  native libraries
cc -O2 -DPREC=z -DBENCH_LIB_accelerate '-DBENCH_CBLAS_HEADER=<Accelerate/Accelerate.h>' \
    -Wno-deprecated-declarations $SRC/bench/gemm/benchx.c -framework Accelerate \
    -o bin/bench-accelerate || fail $? "link accelerate"
OB=$(brew --prefix openblas)
cc -O2 -DPREC=z -DBENCH_LIB_openblas -I$OB/include $SRC/bench/gemm/benchx.c \
    -L$OB/lib -Wl,-rpath,$OB/lib -lopenblas -o bin/bench-openblas || fail $? "link openblas"
c++ -std=c++14 -O3 -mcpu=native -DPREC=z $(pkg-config --cflags eigen3) \
    $SRC/bench/gemm/bench-eigen.cpp -o bin/bench-eigen || fail $? "build eigen"
VARIANTS="$VARIANTS accelerate openblas eigen"
for v in $VARIANTS; do bin/bench-$v -i > logs/identity-$v.txt 2>&1; done
shasum -a 256 bin/bench-* > logs/programs.sha256

{
    echo "date $(date '+%Y-%m-%dT%H:%M:%S%z')"
    sw_vers; uname -a; sysctl -n machdep.cpu.brand_string
    sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize
    cc --version | head -1; brew list --versions openblas eigen
    echo "variants:$VARIANTS; once: $ONCE; rounds $ROUNDS; samples $SAMPLES"
    echo "loadmax $LOADMAX; pause $PAUSE"
    echo "sizes $(echo $SIZES | tr '\n' ' ')"
    echo "status: $(status)"
} > logs/system.txt 2>&1

once() { case " $ONCE " in *" $1 "*) return 0;; esac; return 1; }
shuffle() {     # shuffle SEED WORDS...: the words in a random order
    seed=$1; shift
    echo "$@" | tr ' ' '\n' | grep . \
        | awk -v s=$seed 'BEGIN { srand(s) } { print rand() "\t" $0 }' \
        | sort -n | cut -f2 | tr '\n' ' '
}
echo "start $(date '+%Y-%m-%dT%H:%M:%S%z')" >> $LOG
for r in $(seq 1 $ROUNDS); do
    for n in $SIZES; do
        echo "$(date +%T) round $r n $n: $(status)" >> $LOG
        for v in $(shuffle $((r*100000+n)) $VARIANTS); do
            if [ $r -gt 1 ] && once $v; then continue; fi
            measure res/$v-$r-$n.txt bin/bench-$v $SAMPLES $n || true
        done
    done
    echo "$(date +%T) round $r done" >> $LOG
done
echo "wrong results: $(cat res/*.txt | grep -c WRONG)" >> $LOG
echo "discarded runs: $(cat $LOG.discards 2>/dev/null | wc -l)" >> $LOG
echo "end $(date '+%Y-%m-%dT%H:%M:%S%z')" >> $LOG
if [ -s $LOG.failures ]; then cat $LOG.failures >> $LOG; else echo "overall: all commands succeeded" >> $LOG; fi
