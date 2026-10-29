#!/bin/bash
#  cuda-gpu.sbatch TARBALL TAG CUDA_ARCH: build cuda/ on a GPU node, run
#  make check, make check-memcheck and make bench; logs in $W/cuda/TAG/
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
{
    echo "host $(hostname) job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION date $(date -Iseconds)"
    echo "tarball $(sha256sum $TARBALL)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv
    nvidia-smi -L
    nvcc --version | tail -2
    g++ --version | head -1
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
} > env.txt 2>&1
make CUDA_ARCH=$ARCH > build.log 2>&1; echo "build exit $?" >> env.txt
make check CUDA_ARCH=$ARCH > check.log 2>&1; echo "check exit $?" >> env.txt
make check-memcheck CUDA_ARCH=$ARCH > memcheck.log 2>&1; echo "memcheck exit $?" >> env.txt
make bench CUDA_ARCH=$ARCH BENCH_ARGS="--reps 10 --warmup 2" > bench.log 2>&1; echo "bench exit $?" >> env.txt
make bench CUDA_ARCH=$ARCH BENCH_ARGS="--reps 10 --warmup 2 --pad 7 --shapes '1000,1000,1000 2047,2049,1023'" > bench-pad.log 2>&1; echo "bench-pad exit $?" >> env.txt
date -Iseconds >> env.txt
