#!/bin/bash
#  cuda-step14-final.sbatch TARBALL TAG CUDA_ARCH MAGMA: build cuda/ from scratch
#  with MAGMA on a GPU node; make check, make check-memcheck, the benchmark
#  of all implementations (m = n = k = 50, 100, ..., 4000); then the tests
#  and memcheck of step 14 built with the settings of its experiments;
#  logs in $W/cuda/TAG/cuda/
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
cuobjdump -sass build/step14/gemm.o > sass-step14.txt
#  the experiments of step 14 must pass the tests too
for spec in stages3=-DCULMBLAS_STEP14_STAGES_SMALL=3,-DCULMBLAS_STEP14_STAGES_MEDIUM=3,-DCULMBLAS_STEP14_STAGES_LARGE=3 \
            stages4=-DCULMBLAS_STEP14_STAGES_SMALL=4,-DCULMBLAS_STEP14_STAGES_MEDIUM=4,-DCULMBLAS_STEP14_STAGES_LARGE=4 \
            panels32=-DCULMBLAS_STEP14_PANEL_SMALL=32,-DCULMBLAS_STEP14_PANEL_MEDIUM=32,-DCULMBLAS_STEP14_PANEL_LARGE=32,-DCULMBLAS_STEP14_MIN_BLOCKS=4 \
            panels16=-DCULMBLAS_STEP14_PANEL_LARGE=16,-DCULMBLAS_STEP14_MIN_BLOCKS=4; do
    name=${spec%%=*}; F=${spec#*=}; F=${F//,/ }
    make clean > /dev/null
    make $M NVCCFLAGS="-O3 $F" build/test_gemm > build-$name.log 2>&1; echo "build-$name exit $?" >> env.txt
    build/test_gemm --variants step14 > check-$name.log 2>&1; echo "check-$name exit $?" >> env.txt
    compute-sanitizer --tool memcheck --error-exitcode 1 build/test_gemm --quick --quiet --variants step14 > memcheck-$name.log 2>&1; echo "memcheck-$name exit $?" >> env.txt
done
date -Iseconds >> env.txt
