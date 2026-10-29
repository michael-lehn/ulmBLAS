#!/bin/bash
#  cuda-try-shapes.sbatch TARBALL TAG CUDA_ARCH VARIANTS SHAPES SPEC...: as
#  cuda-try.sbatch with the shapes given (quoted list "m,n,k ...")
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3; VARIANTS=$4; SH=$5; shift 5
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
module purge
module load devel/cuda/12.8
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; } > $K/env.txt 2>&1
for spec in "$@"; do
    name=${spec%%=*}; F="-O3 ${spec#*=}"; F=${F//,/ }
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
