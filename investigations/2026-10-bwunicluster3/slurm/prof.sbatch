#!/bin/bash
#  prof.sbatch ARCH REV: where the time of ulmBLAS DGEMM goes with the
#  OpenBLAS-derived kernel and with BLIS skx 16x14 (packing, macro kernel,
#  micro kernel alone on packed panels), compared with native OpenBLAS, on
#  one exclusive node.  Uses prof.cc of campaign
#  benchmark_results/003-2026-10-25-macbook-m5-profile (unchanged; copied to
#  $W/src/bench/prof.cc with packvar.h).  Output: $W/ARCH/prof-JOBID/.
#
#      sbatch -p cpu_il -J prof-icelake prof.sbatch icelake c0d9c06
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=32000mb
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1; REV=$2
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
A=$W/$ARCH; O=$A/prof-$SLURM_JOB_ID; B=$W/src/bench
mkdir -p $O/res $O/logs && cd $O || exit 1
module purge
module load compiler/gnu/14.2
FULL=$(git -C $W/src/ulmBLAS rev-parse $REV) || exit 1
mkdir src && git -C $W/src/ulmBLAS archive $FULL | tar xf - -C src || exit 1
F="-std=c++17 -O3 -fomit-frame-pointer -Isrc -mavx -mfma -mavx512f"
{
  echo "host $(hostname) job $SLURM_JOB_ID $(date -Iseconds) ulmBLAS $FULL"
  g++ $F -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12 $B/prof.cc -o prof-openblas-kernel && echo built prof-openblas-kernel
  g++ $F -DULMBLAS_DGEMM_UKERNEL_blis_skx_16x14 $B/prof.cc -o prof-skx && echo built prof-skx
  g++ $F -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12 -DWITH_CBLAS -I$A/refs/openblas/include $B/prof.cc $A/refs/openblas/lib/libopenblas.a -o prof-cblas-openblas && echo built prof-cblas-openblas
  sha256sum prof-*
} > logs/build.txt 2>&1
CPU=3; NODE=$(lscpu -p=CPU,NODE | awk -F, -v c=$CPU '$1==c {print $2}')
PIN="numactl --physcpubind=$CPU --membind=$NODE"
echo "start $(date -Iseconds) load $(cat /proc/loadavg)" > logs/run.log
for n in 1000 1536 2000 4000; do
    for r in 1 2; do
        $PIN ./prof-openblas-kernel -b 192,384,8640 -m lib,inst,nopack,pack,kern $n $n $n > res/openblas-kernel-$n-$r.txt 2>&1
        $PIN ./prof-skx -b 224,256,3752 -m lib,inst,nopack,pack,kern $n $n $n > res/skx-$n-$r.txt 2>&1
        $PIN ./prof-cblas-openblas -m cblas $n $n $n > res/cblas-openblas-$n-$r.txt 2>&1
    done
    ./prof-openblas-kernel -b 192,384,8640 -m check $n $n $n > res/check-openblas-kernel-$n.txt 2>&1
    echo "$(date -Iseconds) n=$n done load $(cut -d' ' -f1 /proc/loadavg)" >> logs/run.log
done
echo "end $(date -Iseconds)" >> logs/run.log
