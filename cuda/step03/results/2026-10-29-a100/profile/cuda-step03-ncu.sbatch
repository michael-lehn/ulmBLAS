#!/bin/bash
#  cuda-step03-ncu.sbatch TARBALL TAG CUDA_ARCH: step 03 with the thread
#  blocks 2x4, 4x4, 8x4, 8x8: registers and spills (ptxas -v) and Nsight
#  Compute metrics with stall reasons of one call at n = 2048
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
tar xzf $TARBALL -C $K 2>/dev/null || exit 1
cd $K/cuda || exit 1
module purge
module load devel/cuda/12.8
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; nvidia-smi -L; nvcc --version | tail -1; ncu --version | tail -1; } > env.txt 2>&1
MET=launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,l1tex__throughput.avg.pct_of_peak_sustained_active,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_lg_throttle.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio,smsp__average_warp_latency_issue_stalled_wait.ratio,smsp__average_warp_latency_issue_stalled_not_selected.ratio,smsp__average_warp_latency_issue_stalled_selected.ratio,smsp__average_warp_latency_issue_stalled_dispatch_stall.ratio,smsp__inst_executed.sum,smsp__inst_executed_pipe_fp64.sum
for rc in 2x4 4x4 8x4 8x8; do
    r=${rc%x*}; c=${rc#*x}
    make BUILD=build-$rc CUDA_ARCH=$ARCH NVCCFLAGS="-O3 -Xptxas -v -DCULMBLAS_STEP03_THREAD_ROWS=$r -DCULMBLAS_STEP03_THREAD_COLS=$c" > build-$rc.log 2>&1
    ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 --metrics $MET \
        build-$rc/bench_gemm --reps 1 --warmup 2 --variants step03 --shapes 2048,2048,2048 > ncu-$rc.log 2>&1
    echo "$rc exit $?" >> env.txt
done
date -Iseconds >> env.txt
