#!/bin/bash
#  cuda-prof14b.sbatch TARBALL TAG CUDA_ARCH SPEC...: Nsight Compute
#  (--set full, fixed clock) of step 14 built with SPEC = NAME=FLAG,... at
#  m = n = k = 1024, 2048, 4000 (reports NAME-N.ncu-rep)
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
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; } > $K/env.txt 2>&1
for spec in "$@"; do
    name=${spec%%=*}; F="-O3 ${spec#*=}"; F=${F//,/ }
    D=$K/$name; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" build/bench_gemm > $K/$name.build.log 2>&1
    echo "$name flags $F build exit $?" >> $K/env.txt
    for n in 1024 2048 4000; do
        ncu --set full --import-source no --kernel-name regex:dgemm --launch-skip 2 --launch-count 1 -f -o $K/$name-$n build/bench_gemm --reps 1 --warmup 2 --variants step14 --shapes $n,$n,$n > $K/$name-$n.ncu.log 2>&1
        echo "$name ncu $n exit $?" >> $K/env.txt
    done
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
