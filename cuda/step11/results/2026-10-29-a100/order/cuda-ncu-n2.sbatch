#!/bin/bash
#  cuda-ncu-n2.sbatch TARBALL TAG CUDA_ARCH SIZE...: Nsight Compute for steps
#  10 and 11 at m = n = k = SIZE: time, DRAM, L2, Tensor
#  Cores, stall reasons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3; shift 3
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
tar xzf $TARBALL -C $K 2>/dev/null || exit 1
cd $K/cuda || exit 1
module purge
module load devel/cuda/12.8
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; } > env.txt 2>&1
make -j8 CUDA_ARCH=$ARCH build/bench_gemm > build.log 2>&1; echo "build exit $?" >> env.txt
MET=gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio,smsp__average_warp_latency_issue_stalled_wait.ratio,smsp__inst_executed.sum,sm__cycles_elapsed.avg.per_second
for n in "$@"; do
    ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 --metrics $MET build/bench_gemm --reps 1 --warmup 2 --variants step10 --shapes $n,$n,$n > ncu-step10-$n.log 2>&1
    ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 --metrics $MET build/bench_gemm --reps 1 --warmup 2 --variants step11 --shapes $n,$n,$n > ncu-step11-$n.log 2>&1
    echo "ncu $n done" >> env.txt
done
date -Iseconds >> env.txt
