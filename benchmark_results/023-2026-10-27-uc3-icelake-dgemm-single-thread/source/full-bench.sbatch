#!/bin/bash
#  full-bench.sbatch ARCH: the full single-threaded DGEMM campaign on one
#  exclusive compute node of bwUniCluster 3.0: square m = n = k = 50, 100,
#  ..., 4000 (80 sizes), column major, alpha = 1, beta = 1, for every
#  program in VARIANTS (bench-V: bench2.c linked with a BLAS library, or
#  bench-eigen.cpp), in ROUNDS rounds with a rotated order of the variants.
#  Timing as in kernels-bench.sbatch (bench2.c: first result checked
#  against a naive product up to 2e8 multiply-adds, i.e. n <= 550, warm-up
#  of at least 0.3 s, 7 samples of at least 0.2 s, median); one process at
#  a time, bound to core CPU and its NUMA node with numactl; every library
#  limited to one thread through its environment variable (OpenBLAS and
#  BLIS are built single threaded, MKL is linked sequential, Eigen is built
#  without OpenMP).  Afterwards verify-V checks every size n >= 600 (2000
#  random elements of C against long double sums).  Records the conditions
#  of every measurement, the system, the environment and the build logs of
#  the programs.  Results: $W/ARCH/frun-JOBID/{res,logs}.  NMAX
#  (environment, default 4000) limits the sizes for a test of the script.
#
#      VARIANTS="ulmblas-openblas-skylakex_16x12-it3 openblas mkl eigen" \
#          sbatch -p cpu_il -J fbench-icelake full-bench.sbatch icelake
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=32000mb
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

set -u
#  the C++ programs (ulmBLAS, Eigen) need the libstdc++ of GCC 14.2
module purge
module load compiler/gnu/14.2
ARCH=$1
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
CPU=${CPU:-3}; ROUNDS=${ROUNDS:-3}
NMAX=${NMAX:-4000}   # smaller only for a test of the script
VARIANTS=${VARIANTS:?VARIANTS must be set}
A=$W/$ARCH; O=$A/frun-$SLURM_JOB_ID; L=$O/logs
mkdir -p $O/res $L && cd $O || exit 1
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
#  If CPU is not part of the allocation (e.g. a one-socket allocation on
#  the other socket, see kernels-bench-socket.sbatch), use the 4th CPU of
#  the allocation instead.
#  one thread in every library (no effect where the library is single
#  threaded anyway)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
ALLOWED=$(numactl --show | awk '$1=="physcpubind:" {$1=""; print}')
case " $ALLOWED " in
    *" $CPU "*) ;;
    *) CPU=$(echo $ALLOWED | awk '{print $4}');;
esac
NODE=$(lscpu -p=CPU,NODE | awk -F, -v c=$CPU '$1==c {print $2}')
PIN="numactl --physcpubind=$CPU --membind=$NODE"

for v in $VARIANTS; do
    cp $A/bin/bench-$v $A/bin/verify-$v . || fail $? "copy $v"
done
cp $A/bin/peak . || fail $? "copy peak"
sha256sum bench-* verify-* peak > $L/programs.sha256

{
    echo "host $(hostname) job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION date $(date -Iseconds)"
    echo "cpu $CPU numa node $NODE: $PIN"
    grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo; uname -r
    echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor) driver $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_driver)"
    echo "no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null) smt $(cat /sys/devices/system/cpu/smt/control)"
    echo "thp $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    echo "modules: $(module -t list 2>&1 | tr '\n' ' ')"
    echo "allowed cpus: $(grep Cpus_allowed_list /proc/self/status | cut -f2)"
    echo "allocation: $SLURM_JOB_CPUS_PER_NODE CPUs; exclusive: $(scontrol show job $SLURM_JOB_ID | grep -o 'OverSubscribe=[A-Z]*')"
    echo "sockets of the allocated CPUs: $(for c in $ALLOWED; do lscpu -p=CPU,SOCKET | awk -F, -v c=$c '$1==c {print $2}'; done | sort -u | tr '\n' ' ')"
    echo "measured CPU $CPU, NUMA node $NODE; sizes 50 ... $NMAX; rounds $ROUNDS"
    echo "load $(cat /proc/loadavg); processes of other users: $(ps -eo user= | grep -vcE "^(root|$USER)$")"
    numactl -H | head -3
    echo "== environment"; env | grep -E '^(OMP|OPENBLAS|MKL|BLIS|GOMP|KMP)_' | sort
    echo "== lscpu"; lscpu
} > $L/system.txt 2>&1
#  how the programs were built: reference libraries (build.sbatch) and
#  ulmBLAS (kernels-build.sbatch); kernels.txt lists the kernels each
#  library selects on this CPU type
mkdir -p $L/build
for f in kernels.txt build-env.txt compile-commands.txt build-programs.log; do
    cp $A/logs/$f $L/build/ 2>/dev/null
done
for d in $A/kernels-*; do
    t=${d##*/kernels-}
    for v in $VARIANTS; do
        case $v in ulmblas-*-$t)
            mkdir -p $L/build/kernels-$t
            cp $d/logs/build-env.txt $d/logs/compile-commands.txt $d/logs/programs.sha256 $L/build/kernels-$t/ 2>/dev/null;;
        esac
    done
done

SIZES=$(for n in $(seq 50 50 $NMAX); do echo -n "$n $n $n "; done)
VSIZES=$(seq 600 50 $NMAX)
rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }

HWMON=
for h in /sys/class/hwmon/hwmon*; do
    case "$(cat $h/name 2>/dev/null)" in coretemp|k10temp) HWMON=$h; break;; esac
done
cond() {   # cond KIND VARIANT ROUND: conditions before a measurement
    t=; [ -n "$HWMON" ] && t=$(($(cat $HWMON/temp1_input 2>/dev/null || echo 0)/1000))
    f=$(($(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_cur_freq)/1000))
    echo "$(date +%s),$1,$2,$3,$(cut -d' ' -f1 /proc/loadavg),$t,$f" >> $L/conditions.csv
}
echo "time,kind,variant,round,load1,temp_C,cpu_MHz" > $L/conditions.csv

echo "start $(date -Iseconds)" >> $L/run.log
#  FMA peak of the core (ymm and zmm), before and after the campaign
$PIN ./peak > res/peak-before.txt 2>&1 || fail $? "peak before"
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        cond main $v $r; $PIN ./bench-$v 7 1 $SIZES > res/main-$v-$r.txt || fail $? "main $v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
for v in $VARIANTS; do
    cond verify $v 0; $PIN ./verify-$v $VSIZES > res/verify-$v.txt 2>&1 || fail $? "verify $v"
done
$PIN ./peak > res/peak-after.txt 2>&1 || fail $? "peak after"
echo "wrong results: $(cat res/main-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
rm -f bench-* verify-* peak
