#!/bin/bash
#  cuda-prof14.sbatch TARBALL TAG CUDA_ARCH: what limits step 13?
#  1. Nsight Compute (--set full, fixed clock) of step 13 and cuBLAS at
#     m = n = k = 512, 1024, 2048, 4000 (reports rep-N.ncu-rep);
#  2. SM clock and power (nvidia-smi every 50 ms) while step 13 or cuBLAS
#     runs alone for ~3 s (clockrun), in both orders, 10 s pause between
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -u
TARBALL=$1; TAG=$2; ARCH=$3
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench
K=$W/cuda/$TAG
rm -rf $K && mkdir -p $K || exit 1
tar xzf $TARBALL -C $K 2>/dev/null || exit 1
cd $K/cuda || exit 1
module purge
module load devel/cuda/12.8
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvidia-smi -q -d POWER,CLOCK | grep -E "Power Limit|Max Clocks|SM  |Applications" ; ncu --version | tail -1; } > env.txt 2>&1
make -j8 CUDA_ARCH=$ARCH build/bench_gemm > build.log 2>&1; echo "build exit $?" >> env.txt
nvcc -std=c++17 -arch=$ARCH -I. -O3 -o build/clockrun clockrun.cu build/culmblas/gemm.o build/step*/gemm.o -lcublas >> build.log 2>&1; echo "clockrun build exit $?" >> env.txt
for n in 512 1024 2048 4000; do
    ncu --set full --import-source no --launch-skip 0 --launch-count 6 -f -o rep-$n build/bench_gemm --reps 1 --warmup 2 --variants step13 --shapes $n,$n,$n > ncu-$n.log 2>&1
    echo "ncu $n exit $?" >> env.txt
done
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 50 > clocks.csv &
P=$!
sleep 5
for nr in 1024:15000 2048:2500 4000:330; do
    n=${nr%%:*}; r=${nr#*:}
    for order in "step13 cuBLAS" "cuBLAS step13"; do
        for v in $order; do
            sleep 10
            build/clockrun $v $n $r >> clockrun.log 2>&1
        done
    done
done
sleep 5
kill $P
date -Iseconds >> env.txt
