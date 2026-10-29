#!/bin/bash
#  cuda-try.sbatch TARBALL TAG CUDA_ARCH VARIANTS SPEC...: test variants of
#  a step; SPEC = NAME=FLAG,FLAG,... (compile flags, e.g.
#  w11=-DCULMBLAS_STEP06_WARPS_M=1,-DCULMBLAS_STEP06_WARPS_N=1).  Per
#  spec: build (registers), quick tests, benchmark of VARIANTS (comma
#  separated) for m = n = k = 50, 100, 150, 200, 300, ..., 2000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3; VARIANTS=$4; shift 4
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
module purge
module load devel/cuda/12.8
SH="50,50,50 100,100,100 150,150,150"
for s in $(seq 200 100 2000); do SH="$SH $s,$s,$s"; done
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvcc --version | tail -1; } > $K/env.txt 2>&1
for spec in "$@"; do
    name=${spec%%=*}; F="-O3 -Xptxas -v ${spec#*=}"; F=${F//,/ }
    D=$K/$name; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    echo "$name flags $F" >> $K/env.txt
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" > $K/$name.build.log 2>&1
    echo "$name build exit $?" >> $K/env.txt
    build/test_gemm --quick --quiet > $K/$name.check.log 2>&1
    echo "$name check exit $?" >> $K/env.txt
    build/bench_gemm --reps 10 --warmup 2 --variants $VARIANTS --shapes "$SH" > $K/$name.bench.log 2>&1
    echo "$name bench exit $?" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
