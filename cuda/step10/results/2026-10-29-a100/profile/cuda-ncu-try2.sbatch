#!/bin/bash
#  cuda-ncu-try.sbatch TARBALL TAG CUDA_ARCH VARIANT SPEC...: Nsight Compute
#  (occupancy, Tensor Cores, time) at n = 2048 of VARIANT for builds with
#  other compile flags; SPEC = NAME=FLAG,FLAG,...
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3; VARIANT=$4; shift 4
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
module purge
module load devel/cuda/12.8
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; } > $K/env.txt 2>&1
MET=gpu__time_duration.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio,smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,smsp__average_warp_latency_issue_stalled_wait.ratio,smsp__average_warp_latency_issue_stalled_not_selected.ratio,smsp__average_warp_latency_issue_stalled_selected.ratio
for spec in "$@"; do
    name=${spec%%=*}; F="-O3 ${spec#*=}"; F=${F//,/ }
    D=$K/$name; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    echo "$name flags $F" >> $K/env.txt
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" build/bench_gemm > $K/$name.build.log 2>&1
    echo "$name build exit $?" >> $K/env.txt
    ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 --metrics $MET build/bench_gemm --reps 1 --warmup 2 --variants $VARIANT --shapes 2048,2048,2048 > $K/$name.ncu.log 2>&1
    echo "$name ncu exit $?" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
