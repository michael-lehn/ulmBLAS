#!/bin/bash
#  cuda-ncu-sizes.sbatch TARBALL TAG CUDA_ARCH SPEC...: per build (SPEC =
#  NAME=FLAG,...): SASS of step 09, quick tests, benchmark of steps 08 and
#  09 for n = 2000, 2048, 2050, 2100, 3000, 3050, 4000, and Nsight Compute
#  (sectors, time) for n = 2048 and 2050
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
MET=gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ldgsts.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active
for spec in "$@"; do
    name=${spec%%=*}; F="-O3 ${spec#*=}"; F=${F//,/ }
    D=$K/$name; mkdir -p $D && tar xzf $TARBALL -C $D 2>/dev/null || exit 1
    cd $D/cuda || exit 1
    echo "$name flags $F" >> $K/env.txt
    make -j8 CUDA_ARCH=$ARCH NVCCFLAGS="$F" > $K/$name.build.log 2>&1
    echo "$name build exit $?" >> $K/env.txt
    cuobjdump -sass build/step09/gemm.o | grep -oE "LDGSTS[A-Z0-9.]*" | sort | uniq -c >> $K/env.txt
    build/test_gemm --quick --quiet > $K/$name.check.log 2>&1
    echo "$name check exit $?" >> $K/env.txt
    build/bench_gemm --reps 10 --warmup 2 --variants step08,step09 --shapes "2000,2000,2000 2048,2048,2048 2050,2050,2050 2100,2100,2100 3000,3000,3000 3050,3050,3050 4000,4000,4000" > $K/$name.bench.log 2>&1
    for v in step08 step09; do for n in 2048 2050; do
        ncu --kernel-name dgemm --launch-skip 2 --launch-count 1 --metrics $MET build/bench_gemm --reps 1 --warmup 2 --variants $v --shapes $n,$n,$n > $K/$name.ncu-$v-$n.log 2>&1
    done; done
    echo "$name ncu done" >> $K/env.txt
    cd $K && rm -rf $D
done
date -Iseconds >> $K/env.txt
