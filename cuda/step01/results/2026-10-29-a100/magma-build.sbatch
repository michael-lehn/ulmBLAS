#!/bin/bash
#  magma-build.sbatch: MAGMA 2.10.0 (sha256 ea0c57fc...) for sm_80 with
#  CUDA 12.8, GCC 11.5 and MKL 2025.0 (make.inc.mkl-gcc), installed into
#  $W/magma/2.10.0-sm80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64000mb
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -u
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
T=$W/up/magma-2.10.0.tar.gz
P=$W/magma/2.10.0-sm80
B=$W/magma/build
module purge
module load devel/cuda/12.8 numlib/mkl/2025.0
echo "sha256 $(sha256sum $T)"
rm -rf $B $P && mkdir -p $B $P && tar xzf $T -C $B || exit 1
cd $B/magma-2.10.0 || exit 1
cp make.inc-examples/make.inc.mkl-gcc make.inc
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; nvcc --version | tail -2; gcc --version | head -1; echo "MKLROOT=$MKLROOT"; echo "CUDA_HOME=${CUDA_HOME:-}"; } > $P/build-env.txt
time make -j $SLURM_CPUS_PER_TASK lib GPU_TARGET=sm_80 CUDADIR=$(dirname $(dirname $(which nvcc))) > $P/build.log 2>&1
echo "make lib exit $?" >> $P/build-env.txt
make install prefix=$P GPU_TARGET=sm_80 CUDADIR=$(dirname $(dirname $(which nvcc))) >> $P/build.log 2>&1
echo "make install exit $?" >> $P/build-env.txt
cp make.inc $P/make.inc.used
ls -la $P/lib $P/include | head -20 >> $P/build-env.txt
