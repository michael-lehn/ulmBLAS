#!/bin/bash
#  cuda-step04-shapes.sbatch TARBALL TAG CUDA_ARCH SPEC...: the final code
#  of step 04 with other blocks and passes; SPEC = RxC-pP (block per
#  thread, values of l per pass).  Per spec: build (registers from ptxas),
#  quick tests, benchmark of step 04 for m = n = k = 50, 100, ..., 4000
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
module purge
module load devel/cuda/12.8
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvcc --version | tail -1; } > $K/env.txt 2>&1
for spec in "$@"; do
    IFS=- read rc p <<< "$spec"
    r=${rc%x*}; c=${rc#*x}
    D=$K/$spec; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    F="-O3 -Xptxas -v -DCULMBLAS_STEP04_THREAD_ROWS=$r -DCULMBLAS_STEP04_THREAD_COLS=$c -DCULMBLAS_STEP04_PASS=${p#p}"
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" > $K/$spec.build.log 2>&1
    echo "$spec build exit $?" >> $K/env.txt
    grep -A2 'step04.*dgemm' $K/$spec.build.log | grep -E 'registers|spill' >> $K/env.txt
    build/test_gemm --quick --quiet > $K/$spec.check.log 2>&1
    echo "$spec check exit $?" >> $K/env.txt
    build/bench_gemm --reps 10 --warmup 2 --variants step04 > $K/$spec.bench.log 2>&1
    echo "$spec bench exit $?" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
