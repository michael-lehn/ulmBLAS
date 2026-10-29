#!/bin/bash
#  cuda-clocks.sbatch TARBALL TAG CUDA_ARCH SPEC...: per build (SPEC =
#  NAME=FLAG,...) the benchmark of VARIANTS at n = 2000 and 4000 with 100
#  timed calls, while nvidia-smi logs SM clock, power and throttle reasons
#  every 100 ms; and Nsight Compute names of the cuBLAS kernels
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
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvidia-smi -q -d POWER,CLOCK | head -60; } > $K/env.txt 2>&1
for spec in "$@"; do
    name=${spec%%=*}; F="-O3 ${spec#*=}"; F=${F//,/ }
    D=$K/$name; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" build/bench_gemm > $K/$name.build.log 2>&1
    for v in step10 cuBLAS; do for n in 2000 4000; do
        vv=$v; [ $v = cuBLAS ] && vv=step01
        [ $v = cuBLAS ] && [ $name != g0 ] && continue
        nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 100 > $K/$name-$v-$n.smi.csv &
        P=$!
        sleep 1
        build/bench_gemm --reps 100 --warmup 5 --variants $vv --shapes $n,$n,$n > $K/$name-$v-$n.bench.log 2>&1
        sleep 1
        kill $P
    done; done
    cd $K && rm -rf $D
done
cd $K && mkdir -p kn && tar xzf $TARBALL -C kn 2>/dev/null && cd kn/cuda && make -j8 CUDA_ARCH=$ARCH build/bench_gemm > /dev/null 2>&1 && ncu --launch-count 40 --metrics gpu__time_duration.sum,sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,lts__t_sector_hit_rate.pct build/bench_gemm --reps 1 --warmup 1 --variants step10 --shapes "2048,2048,2048 4000,4000,4000" > $K/ncu-names.log 2>&1
date -Iseconds >> $K/env.txt
