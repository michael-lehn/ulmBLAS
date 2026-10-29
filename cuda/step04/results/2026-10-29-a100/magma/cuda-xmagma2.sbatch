#!/bin/bash
#  cuda-xmagma.sbatch TARBALL TAG CUDA_ARCH MAGMA: the MAGMA-like kernels
#  xm0..xm4 against magmablas, step03, step04 (4x4, passes of 4, no k
#  checks); quick tests; benchmark n = 50, 100, 150, 200, 300, ..., 2000;
#  Nsight Compute metrics at n = 1000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000mb
#SBATCH --time=00:30:00
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
SH="50,50,50 100,100,100 150,150,150"
for s in $(seq 200 100 2000); do SH="$SH $s,$s,$s"; done
{ echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; echo "tarball $(sha256sum $TARBALL)"; nvidia-smi -L; nvcc --version | tail -1; } > env.txt 2>&1
F="-O3 -Xptxas -v"
make -j8 CUDA_ARCH=$ARCH MAGMA=$MAGMA NVCCFLAGS="$F" > build.log 2>&1; echo "build exit $?" >> env.txt
grep -A2 -E "xmagma" build.log | grep -E 'Compiling|registers|spill' >> env.txt
build/test_gemm --quick --quiet > check.log 2>&1; echo "check exit $?" >> env.txt
build/bench_gemm --reps 10 --warmup 2 --variants magmablas,xm1,xm5,xm6,xm7,xm8,xm9,xm10,xm11,xm12,xm13 --shapes "$SH" > bench.log 2>&1; echo "bench exit $?" >> env.txt
MET=launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,l1tex__throughput.avg.pct_of_peak_sustained_active,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,smsp__inst_executed.sum,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,gpu__time_duration.sum
for v in xm1 xm5 xm6 xm7 xm8 xm9 xm10 xm11 xm12 xm13; do
    ncu --kernel-name regex:"dgemm|gemm_template" --launch-skip 2 --launch-count 1 --metrics $MET build/bench_gemm --reps 1 --warmup 2 --variants $v --shapes 1000,1000,1000 > ncu-$v.log 2>&1
    echo "ncu $v exit $?" >> env.txt
done
date -Iseconds >> env.txt
