#!/bin/bash
#  cuda-gpu-magma.sbatch TARBALL TAG CUDA_ARCH MAGMA: build cuda/ from
#  scratch with MAGMA on a GPU node, run make check, make check-memcheck and
#  the benchmarks of step 01; logs in $W/cuda/TAG/cuda/
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
    echo "MAGMA $MAGMA: $(head -1 $MAGMA/build-env.txt 2>/dev/null)"
    echo "make settings: $M"
} > env.txt 2>&1
make clean > /dev/null
make $M > build.log 2>&1; echo "build exit $?" >> env.txt
make check $M > check.log 2>&1; echo "check exit $?" >> env.txt
make check-memcheck $M > memcheck.log 2>&1; echo "memcheck exit $?" >> env.txt
#  default shapes of bench_gemm: m = n = k = 100, 200, ..., 4000
#SQ=$(seq -s " " 100 100 4000 | awk '{for (i = 1; i <= NF; ++i) printf "%s%d,%d,%d", (i > 1 ? " " : ""), $i, $i, $i}')
RECT="4096,4096,256 4096,256,4096 256,4096,4096 1000,1000,1000 1500,700,3000"
build/bench_gemm --reps 10 --warmup 2 > bench-square.log 2>&1; echo "bench-square exit $?" >> env.txt
build/bench_gemm --reps 10 --warmup 2 --shapes "$RECT" > bench-rect.log 2>&1; echo "bench-rect exit $?" >> env.txt
build/bench_gemm --reps 10 --warmup 2 --pad 7 --shapes "1000,1000,1000 2047,2049,1023" > bench-pad.log 2>&1; echo "bench-pad exit $?" >> env.txt
date -Iseconds >> env.txt
