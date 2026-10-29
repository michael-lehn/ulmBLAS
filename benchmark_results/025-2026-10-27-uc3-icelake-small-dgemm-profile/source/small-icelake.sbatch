#!/bin/bash
#  small-icelake.sbatch REV: where ulmBLAS DGEMM spends its time for small
#  matrices on Ice Lake (bwUniCluster 3.0 cpu_il), and which code path native
#  OpenBLAS 0.3.34 (SKYLAKEX) takes there.  One exclusive node, core 3 and
#  its NUMA node (numactl), one process at a time.
#
#  1. sweep: bench2 (campaign 023: first result checked, warm-up >= 0.3 s,
#     7 samples >= 0.2 s, median), square n in SIZES, for ulmBLAS REV with
#     OpenBLAS's SKYLAKEX kernel (as in campaign 023) and native OpenBLAS;
#     ROUNDS rounds, order alternating.  OpenBLAS uses its small-matrix
#     kernel (no packing) for NN with m*n*k <= 1e6
#     (kernel/x86_64/dgemm_small_kernel_permit_skylakex.c), i.e. up to
#     n = 100: the sweep crosses n = 100/101.
#  2. profile: prof-ob.cc of campaign 016 (source of this job directory)
#     with the SKYLAKEX kernel: modes lib, inst (time in packing B, A and
#     macro kernel), nopack, pack, kern, pool, and cblas (native OpenBLAS),
#     block sizes 192, 384, 8640, for n in PROFN; ROUNDS rounds.
#
#      sbatch -p cpu_il -J small-icelake small-icelake.sbatch df58cfa
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=16000mb
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -u
module purge
module load compiler/gnu/14.2
REV=$1
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
S=$W/small; R=$W/icelake/refs/openblas; D=$S/source
O=$S/run-$SLURM_JOB_ID; L=$O/logs
mkdir -p $O/res $L && cd $O || exit 1
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
ROUNDS=${ROUNDS:-3}
SIZES=${SIZES:-"8 16 24 32 40 48 50 56 64 72 80 88 96 98 99 100 101 102 104 112 120 128 136 150 160 176 192 200 224 250"}
PROFN=${PROFN:-"48 96 100 101 144 192 240"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CPU=3; NODE=$(lscpu -p=CPU,NODE | awk -F, -v c=$CPU '$1==c {print $2}')
PIN="numactl --physcpubind=$CPU --membind=$NODE"

FULL=$(git -C $S/ulmBLAS rev-parse $REV) || exit 1
mkdir src && git -C $S/ulmBLAS archive $FULL | tar xf - -C src || exit 1
K="-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12"
{
  echo "host $(hostname) job $SLURM_JOB_ID $(date -Iseconds) ulmBLAS $FULL"
  (cd src && make -j16 cblas ARCHFLAGS="$K") > $L/build-ulmblas.log 2>&1 && echo built ulmBLAS
  cat src/interfaces/blas/C/build.cblas/compile.cmd
  gcc -O2 -Isrc $D/bench2.c src/libcblas.a -lstdc++ -lm -o bench-ulmblas && echo built bench-ulmblas
  gcc -O2 -I$R/include $D/bench2.c $R/lib/libopenblas.a -lm -o bench-openblas && echo built bench-openblas
  F="-std=c++17 -O3 -fomit-frame-pointer -Isrc $K"
  g++ $F $D/prof-ob.cc -o prof && echo built prof
  g++ $F -DWITH_CBLAS -I$R/include $D/prof-ob.cc $R/lib/libopenblas.a -o prof-cblas && echo built prof-cblas
  cat > corename.c <<'EOF'
#include <stdio.h>
char *openblas_get_corename(void);
char *openblas_get_config(void);
int main(void) { printf("corename %s\nconfig %s\n", openblas_get_corename(), openblas_get_config()); return 0; }
EOF
  gcc corename.c $R/lib/libopenblas.a -lm -o corename && ./corename
  nm $R/lib/libopenblas.a 2>/dev/null | grep -E ' T dgemm_small_kernel_(b0_)?nn' | sort -u
  sha256sum bench-* prof prof-cblas $R/lib/libopenblas.a
} > $L/build.txt 2>&1 || fail $? build
{
  echo "date $(date -Iseconds)"; uname -a; cat /etc/redhat-release
  gcc --version | head -1; lscpu | grep -E 'Model name|Socket|NUMA node'
  grep -m1 microcode /proc/cpuinfo
  echo "cpu $CPU node $NODE; governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor 2>/dev/null)"
  echo "load $(cat /proc/loadavg)"; echo "sizes $SIZES"; echo "profn $PROFN"; echo "rounds $ROUNDS"
} > $L/system.txt 2>&1

ARGS=""; for n in $SIZES; do ARGS="$ARGS $n $n $n"; done
echo "start $(date -Iseconds)" > $L/run.log
for r in $(seq 1 $ROUNDS); do
    if [ $((r%2)) = 1 ]; then V="ulmblas openblas"; else V="openblas ulmblas"; fi
    for v in $V; do
        $PIN ./bench-$v 7 1 $ARGS > res/sweep-$v-$r.txt || fail $? "sweep $v $r"
        echo "$(date -Iseconds) sweep round $r $v done" >> $L/run.log
    done
done
for n in $PROFN; do
    ./prof -b 192,384,8640 -m check $n $n $n > res/check-$n.txt 2>&1 || fail $? "check $n"
done
for r in $(seq 1 $ROUNDS); do
    for n in $PROFN; do
        $PIN ./prof -b 192,384,8640 -m lib,inst,nopack,pack,kern,pool $n $n $n > res/prof-$n-$r.txt 2>&1 || fail $? "prof $n $r"
        $PIN ./prof-cblas -b 192,384,8640 -m cblas $n $n $n > res/cblas-$n-$r.txt 2>&1 || fail $? "cblas $n $r"
    done
    echo "$(date -Iseconds) prof round $r done" >> $L/run.log
done
echo "wrong results: $(cat res/sweep-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
