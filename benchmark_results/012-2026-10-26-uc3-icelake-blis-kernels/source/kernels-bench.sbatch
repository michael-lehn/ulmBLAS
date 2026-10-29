#!/bin/bash
#  kernels-bench.sbatch ARCH: single-threaded DGEMM campaign for the
#  micro kernels in ulmBLAS (programs bench-ulmblas-KERNEL-TAG of
#  kernels-build.sbatch) and the reference libraries (programs of
#  build.sbatch), on one exclusive compute node of bwUniCluster 3.0.  As
#  bench.sbatch (same programs, timing and statistics), with the sizes of
#  the kernel campaigns: square n = 7 ... 768 and n = 1000 ... 4096 (8
#  sizes, the target range m = n = k >= 1000), the thin shapes, beta = 1,
#  in ROUNDS rounds with a rotated order of the variants.  VARIANTS must be
#  given, e.g.
#
#      VARIANTS="ulmblas-avx-k2 ulmblas-skx_16x14-k2 blis-auto openblas" \
#          sbatch -p cpu_il -J kbench-icelake kernels-bench.sbatch icelake
#
#  The node is allocated exclusively so that no other job shares the L3
#  cache, the memory controllers or the power budget (turbo clock) of the
#  measured core; one process runs at a time, bound to core CPU and its NUMA
#  node with numactl.  Results: $W/ARCH/run-JOBID/{res,logs}.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=32000mb
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

set -u
#  the C++ programs (ulmBLAS, Eigen) need the libstdc++ of GCC 14.2
module purge
module load compiler/gnu/14.2
ARCH=$1
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
CPU=${CPU:-3}; ROUNDS=${ROUNDS:-3}
VARIANTS=${VARIANTS:?VARIANTS must be set}
A=$W/$ARCH; O=$A/krun-$SLURM_JOB_ID; L=$O/logs
mkdir -p $O/res $L && cd $O || exit 1
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
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
    echo "load $(cat /proc/loadavg); processes of other users: $(ps -eo user= | grep -vcE "^(root|$USER)$")"
    numactl -H | head -3
} > $L/system.txt 2>&1

SQ="7 7 7 31 31 31 64 64 64 100 100 100 128 128 128 199 199 199 256 256 256 384 384 384 512 512 512 768 768 768"
LARGE="1000 1000 1000 1024 1024 1024 1500 1500 1500 2000 2000 2000 2048 2048 2048 3000 3000 3000 4000 4000 4000 4096 4096 4096"
THIN="2000 2000 32 2000 2000 128 2000 2000 256 4000 4000 64"
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
        cond main $v $r; $PIN ./bench-$v 7 1 $SQ $LARGE $THIN > res/main-$v-$r.txt || fail $? "main $v $r"
        echo "$(date -Iseconds) round $r $v done, load $(cut -d' ' -f1 /proc/loadavg)" >> $L/run.log
    done
done
for v in $VARIANTS; do
    $PIN ./verify-$v 1000 1500 2000 3000 4000 4096 > res/verify-$v.txt 2>&1 || fail $? "verify $v"
done
$PIN ./peak > res/peak-after.txt 2>&1 || fail $? "peak after"
echo "wrong results: $(cat res/main-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
rm -f bench-* verify-* peak
