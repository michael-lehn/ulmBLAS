#!/bin/bash
#  cuda-step04-code.sbatch TAG CUDA_ARCH TARBALL...: forms of the step 04
#  code (one tarball each): build (registers), SASS of the kernel, quick
#  tests, benchmark of steps 03 and 04 for n = 50, 100, 150, 200, 300, ..., 2000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TAG=$1; ARCH=$2; shift 2
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
module purge
module load devel/cuda/12.8
SH="50,50,50 100,100,100 150,150,150"
for s in $(seq 200 100 2000); do SH="$SH $s,$s,$s"; done
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; nvidia-smi -L; nvcc --version | tail -1; } > $K/env.txt 2>&1
for T in "$@"; do
    v=$(basename $T .tgz)
    echo "tarball $(sha256sum $T)" >> $K/env.txt
    D=$K/$v; mkdir -p $D && tar xzf $T -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="-O3 -Xptxas -v" > $K/$v.build.log 2>&1
    echo "$v build exit $?" >> $K/env.txt
    grep -A2 'step04.*dgemm' $K/$v.build.log | grep -E 'registers|spill' >> $K/env.txt
    cuobjdump -sass build/step04/gemm.o > $K/$v.sass.txt
    echo "$v SASS: $(grep -c 'LDG' $K/$v.sass.txt) LDG, $(grep -c 'DFMA' $K/$v.sass.txt) DFMA" >> $K/env.txt
    build/test_gemm --quick --quiet > $K/$v.check.log 2>&1
    echo "$v check exit $?" >> $K/env.txt
    build/bench_gemm --reps 10 --warmup 2 --variants step03,step04 --shapes "$SH" > $K/$v.bench.log 2>&1
    echo "$v bench exit $?" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
