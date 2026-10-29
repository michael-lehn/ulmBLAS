#!/bin/bash
#  cuda-step05-alt.sbatch TARBALL TAG CUDA_ARCH SPEC...: the final code of
#  step 05 with other settings, m = n = k = 50, 100, ..., 4000; SPEC =
#  tokens separated by '-': bN __launch_bounds__(threads, N), txN
#  ThreadsX, tyN ThreadsY, pnN Panel ('def': unchanged).  Per spec: build
#  (registers), quick tests, benchmark of steps 04 and 05
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
declare -A MAC=([tx]=THREADS_X [ty]=THREADS_Y [pn]=PANEL)
for spec in "$@"; do
    D=$K/$spec; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    F="-O3 -Xptxas -v"
    if [ "$spec" != def ]; then
        IFS=- read -ra toks <<< "$spec"
        for t in "${toks[@]}"; do
            if [ "${t:0:1}" = b ]; then
                sed -i "s/__launch_bounds__(ThreadsX\*ThreadsY, 1)/__launch_bounds__(ThreadsX*ThreadsY, ${t:1})/" step05/gemm.cu
            else
                F="$F -DCULMBLAS_STEP05_${MAC[${t:0:2}]}=${t:2}"
            fi
        done
    fi
    echo "$spec flags $F; $(grep -m1 '__launch_bounds__' step05/gemm.cu)" >> $K/env.txt
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" > $K/$spec.build.log 2>&1
    echo "$spec build exit $?" >> $K/env.txt
    grep -A2 'step05.*dgemm' $K/$spec.build.log | grep -E 'registers|spill' >> $K/env.txt
    build/test_gemm --quick --quiet > $K/$spec.check.log 2>&1
    echo "$spec check exit $?" >> $K/env.txt
    build/bench_gemm --reps 10 --warmup 2 --variants step04,step05 > $K/$spec.bench.log 2>&1
    echo "$spec bench exit $?" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
