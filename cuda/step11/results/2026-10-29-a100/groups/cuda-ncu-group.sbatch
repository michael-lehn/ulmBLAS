#!/bin/bash
#  cuda-ncu-group.sbatch TARBALL TAG CUDA_ARCH SIZE SPEC...: per build
#  (SPEC = NAME=FLAG,...) Nsight Compute of step 11 at m = n = k = SIZE
#  (DRAM, L2, time at the clock of Nsight Compute), and the benchmark with
#  100 calls while nvidia-smi logs the SM clock and the power
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3; N=$4; shift 4
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
module purge
module load devel/cuda/12.8
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; } > $K/env.txt 2>&1
MET=gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg.per_second
for spec in "$@"; do
    name=${spec%%=*}; F="-O3 ${spec#*=}"; F=${F//,/ }
    D=$K/$name; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" build/bench_gemm > $K/$name.build.log 2>&1
    echo "$name flags $F build exit $?" >> $K/env.txt
    ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 --metrics $MET build/bench_gemm --reps 1 --warmup 2 --variants step11 --shapes $N,$N,$N > $K/$name.ncu.log 2>&1
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader -lms 100 > $K/$name.smi.csv &
    P=$!
    sleep 1
    build/bench_gemm --reps 100 --warmup 5 --variants step11 --shapes $N,$N,$N > $K/$name.bench.log 2>&1
    sleep 1
    kill $P
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
