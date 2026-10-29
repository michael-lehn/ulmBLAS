#!/bin/bash
#  cuda-check.sbatch TARBALL TAG CUDA_ARCH MAGMA: build, make check, make
#  check-memcheck; logs in $W/cuda/TAG/cuda/
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:20:00
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
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; } > env.txt 2>&1
make -j8 $M > build.log 2>&1; echo "build exit $?" >> env.txt
make check $M > check.log 2>&1; echo "check exit $?" >> env.txt
make check-memcheck $M > memcheck.log 2>&1; echo "memcheck exit $?" >> env.txt
date -Iseconds >> env.txt
