#!/bin/bash
#  cuda-step05-try.sbatch TARBALL TAG CUDA_ARCH MAGMA SPEC...: test variants
#  of step 05; SPEC = tokens separated by '-' that set the compile-time
#  switches: txN ThreadsX, tyN ThreadsY, trN ThreadRows, tcN ThreadCols,
#  pnN Panel, csN COL_STRIDE, ixN INDEX, mbN MIN_BLOCKS, pdN PAD ('def':
#  the defaults).  Per spec: build (registers), quick tests, benchmark of
#  magmablas, steps 04 and 05 for m = n = k = 50, 100, 150, 200, 300, ..., 2000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3; MAGMA=$4; shift 4
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
module purge
module load devel/cuda/12.8 numlib/mkl/2025.0
SH="50,50,50 100,100,100 150,150,150"
for s in $(seq 200 100 2000); do SH="$SH $s,$s,$s"; done
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvcc --version | tail -1; } > $K/env.txt 2>&1
declare -A MAC=([tx]=THREADS_X [ty]=THREADS_Y [tr]=THREAD_ROWS [tc]=THREAD_COLS [pn]=PANEL [cs]=COL_STRIDE [ix]=INDEX [mb]=MIN_BLOCKS [pd]=PAD)
for spec in "$@"; do
    F="-O3 -Xptxas -v"
    if [ "$spec" != def ]; then
        IFS=- read -ra toks <<< "$spec"
        for t in "${toks[@]}"; do F="$F -DCULMBLAS_STEP05_${MAC[${t:0:2}]}=${t:2}"; done
    fi
    D=$K/$spec; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    echo "$spec flags $F" >> $K/env.txt
    make -j8 CUDA_ARCH=$ARCH MAGMA=$MAGMA NVCCFLAGS="$F" > $K/$spec.build.log 2>&1
    echo "$spec build exit $?" >> $K/env.txt
    grep -A2 'step05.*dgemm' $K/$spec.build.log | grep -E 'registers|spill' >> $K/env.txt
    build/test_gemm --quick --quiet > $K/$spec.check.log 2>&1
    echo "$spec check exit $?" >> $K/env.txt
    build/bench_gemm --reps 10 --warmup 2 --variants magmablas,step04,step05 --shapes "$SH" > $K/$spec.bench.log 2>&1
    echo "$spec bench exit $?" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
