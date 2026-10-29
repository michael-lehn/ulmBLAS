#!/bin/bash
#  cuda-step04-try.sbatch TARBALL TAG CUDA_ARCH SPEC...: test variants of
#  step 04; SPEC = RxC-uU-pP-bB (block per thread, values of l per pass,
#  prefetch 0/1, minimum blocks per SM for __launch_bounds__).  Per spec:
#  build, registers (ptxas), quick tests, benchmark of steps 03 and 04 for
#  m = n = k = 50, 100, 150, 200, 300, 400, ..., 2000
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
SH="50,50,50 100,100,100 150,150,150"
for s in $(seq 200 100 2000); do SH="$SH $s,$s,$s"; done
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvcc --version | tail -1; } > $K/env.txt 2>&1
for spec in "$@"; do
    IFS=- read rc u p b <<< "$spec"
    r=${rc%x*}; c=${rc#*x}
    D=$K/$spec; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    F="-O3 -Xptxas -v -DCULMBLAS_STEP04_THREAD_ROWS=$r -DCULMBLAS_STEP04_THREAD_COLS=$c -DCULMBLAS_STEP04_UNROLL=${u#u} -DCULMBLAS_STEP04_PREFETCH=${p#p} -DCULMBLAS_STEP04_MIN_BLOCKS=${b#b}"
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" > $K/$spec.build.log 2>&1
    echo "$spec build exit $?" >> $K/env.txt
    grep -A2 'step04.*dgemm' $K/$spec.build.log | grep -E 'registers|spill' >> $K/env.txt
    build/test_gemm --quick --quiet > $K/$spec.check.log 2>&1
    echo "$spec check exit $?" >> $K/env.txt
    build/bench_gemm --reps 10 --warmup 2 --variants step03,step04 --shapes "$SH" > $K/$spec.bench.log 2>&1
    echo "$spec bench exit $?" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
