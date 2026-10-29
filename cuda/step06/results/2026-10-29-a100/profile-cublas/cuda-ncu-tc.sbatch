#!/bin/bash
#  cuda-ncu-tc.sbatch TARBALL TAG CUDA_ARCH: Nsight Compute at n = 2048 for
#  every kernel launched by bench_gemm for step05, step06 and cuBLAS:
#  names, time, Tensor Core (DMMA) and FP64 instructions and cycles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:20:00
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
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; ncu --version | tail -1; } > env.txt 2>&1
make -j8 CUDA_ARCH=$ARCH > build.log 2>&1; echo "build exit $?" >> env.txt
cuobjdump -sass build/step06/gemm.o | grep -oE "DMMA[A-Z0-9.]*" | sort | uniq -c > sass-dmma-step06.txt
MET=gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,sm__inst_executed_pipe_tensor_op_dmma.sum,sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fp64.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active
for v in step05 step06; do
    ncu --launch-count 12 --metrics $MET build/bench_gemm --reps 1 --warmup 2 --variants $v --shapes 2048,2048,2048 > ncu-$v.log 2>&1
    echo "ncu $v exit $?" >> env.txt
done
date -Iseconds >> env.txt
