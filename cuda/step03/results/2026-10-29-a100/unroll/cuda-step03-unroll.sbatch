#!/bin/bash
#  cuda-step03-unroll.sbatch TARBALL TAG CUDA_ARCH: experiment for the
#  README of step 03 (not part of the tutorial code): the loop over l with
#  '#pragma unroll U' for the thread blocks 2x4, 4x4, 8x4; per build the
#  loads and multiply-adds of the loop (SASS), the benchmark n = 1000 ...
#  4000 and Nsight Compute metrics at n = 2048
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
module purge
module load devel/cuda/12.8
MET=launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,l1tex__throughput.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvcc --version | tail -1; } > $K/env.txt 2>&1
for spec in 2x4-u1 2x4-u2 4x4-u1 4x4-u2 8x4-u1 8x4-u2; do
    rc=${spec%-u*}; u=${spec#*-u}; r=${rc%x*}; c=${rc#*x}
    D=$K/$spec; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    sed -i 's|^\(        for (std::ptrdiff_t l = 0; l < k; ++l) {\)$|        #pragma unroll '$u'\n\1|' step03/gemm.cu
    grep -B1 -A0 'for (std::ptrdiff_t l = 0; l < k; ++l)' step03/gemm.cu > $K/$spec.loop.txt
    make CUDA_ARCH=$ARCH NVCCFLAGS="-O3 -Xptxas -v -DCULMBLAS_STEP03_THREAD_ROWS=$r -DCULMBLAS_STEP03_THREAD_COLS=$c" > $K/$spec.build.log 2>&1
    cuobjdump -sass build/step03/gemm.o > $K/$spec.sass.txt
    build/bench_gemm --reps 10 --warmup 2 --variants step03 --shapes "1000,1000,1000 2000,2000,2000 3000,3000,3000 4000,4000,4000" > $K/$spec.bench.log 2>&1
    echo "$spec bench exit $?" >> $K/env.txt
    ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 --metrics $MET \
        build/bench_gemm --reps 1 --warmup 2 --variants step03 --shapes 2048,2048,2048 > $K/$spec.ncu.log 2>&1
    echo "$spec ncu exit $?" >> $K/env.txt
done
date -Iseconds >> $K/env.txt
