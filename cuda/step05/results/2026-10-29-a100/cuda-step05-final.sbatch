#!/bin/bash
#  cuda-step05-final.sbatch TARBALL TAG CUDA_ARCH MAGMA: build cuda/ from scratch
#  with MAGMA on a GPU node; make check, make check-memcheck, the benchmark
#  of all implementations (m = n = k = 50, 100, ..., 4000) and an Nsight
#  Compute profile of steps 05 and 04 (n = 2048)
#  at n = 2048; logs in $W/cuda/TAG/cuda/
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3; MAGMA=$4
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
tar xzf $TARBALL -C $K 2>/dev/null || exit 1
cd $K/cuda || exit 1
module purge
module load devel/cuda/12.8 numlib/mkl/2025.0
M="CUDA_ARCH=$ARCH MAGMA=$MAGMA"
{
    echo "host $(hostname) job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION date $(date -Iseconds)"
    echo "tarball $(sha256sum $TARBALL)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv
    nvidia-smi -L
    nvcc --version | tail -2
    g++ --version | head -1
    ncu --version | tail -1
    echo "MAGMA $MAGMA"
    echo "make settings: $M"
} > env.txt 2>&1
make clean > /dev/null
make $M > build.log 2>&1; echo "build exit $?" >> env.txt
make check $M > check.log 2>&1; echo "check exit $?" >> env.txt
make check-memcheck $M > memcheck.log 2>&1; echo "memcheck exit $?" >> env.txt
build/bench_gemm --reps 10 --warmup 2 > bench-square.log 2>&1; echo "bench-square exit $?" >> env.txt
ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 \
    --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,l1tex__throughput.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,smsp__inst_executed.sum,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,launch__occupancy_limit_shared_mem \
    build/bench_gemm --reps 1 --warmup 2 --variants step05 --shapes 2048,2048,2048 > ncu-step05.log 2>&1; echo "ncu-step05 exit $?" >> env.txt
ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 \
    --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,l1tex__throughput.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,smsp__inst_executed.sum,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,launch__occupancy_limit_shared_mem \
    build/bench_gemm --reps 1 --warmup 2 --variants step04 --shapes 2048,2048,2048 > ncu-step04.log 2>&1; echo "ncu-step04 exit $?" >> env.txt
date -Iseconds >> env.txt
