#!/bin/bash
#  prof-ob.sbatch ARCH REV: does calling the OpenBLAS kernel once per tile
#  (ulmBLAS's macro kernel) instead of once per packed block (OpenBLAS's
#  driver) explain the gap to native OpenBLAS?  prof-ob.cc (prof.cc with the
#  modes macro, macronopack, macrocheck) on one exclusive node, for sizes
#  where every block can be passed to the kernel (m % 16 == 0, n % 12 == 0
#  for every block with MC = 192).  Output: $W/ARCH/profob-JOBID/.
#
#      sbatch -p cpu_il -J profob-icelake prof-ob.sbatch icelake c0d9c06
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=32000mb
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1; REV=$2
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
A=$W/$ARCH; O=$A/profob-$SLURM_JOB_ID; B=$W/src/bench
mkdir -p $O/res $O/logs && cd $O || exit 1
module purge
module load compiler/gnu/14.2
FULL=$(git -C $W/src/ulmBLAS rev-parse $REV) || exit 1
mkdir src && git -C $W/src/ulmBLAS archive $FULL | tar xf - -C src || exit 1
F="-std=c++17 -O3 -fomit-frame-pointer -Isrc -mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12"
{
  echo "host $(hostname) job $SLURM_JOB_ID $(date -Iseconds) ulmBLAS $FULL"
  g++ $F $B/prof-ob.cc -o prof-ob && echo built prof-ob
  g++ $F -DWITH_CBLAS -I$A/refs/openblas/include $B/prof-ob.cc $A/refs/openblas/lib/libopenblas.a -o prof-ob-cblas && echo built prof-ob-cblas
  sha256sum prof-ob*
} > logs/build.txt 2>&1
CPU=3; NODE=$(lscpu -p=CPU,NODE | awk -F, -v c=$CPU '$1==c {print $2}')
PIN="numactl --physcpubind=$CPU --membind=$NODE"
echo "start $(date -Iseconds) load $(cat /proc/loadavg)" > logs/run.log
for n in 1536 2016 3072 4032; do
    ./prof-ob -b 192,384,8640 -m macrocheck $n $n $n > res/mcheck-$n.txt 2>&1
    for r in 1 2 3; do
        $PIN ./prof-ob -b 192,384,8640 -m lib,inst,nopack,macro,macronopack,kern $n $n $n > res/ob-$n-$r.txt 2>&1
        $PIN ./prof-ob-cblas -m cblas $n $n $n > res/cblas-$n-$r.txt 2>&1
    done
    echo "$(date -Iseconds) n=$n done load $(cut -d' ' -f1 /proc/loadavg)" >> logs/run.log
done
echo "end $(date -Iseconds)" >> logs/run.log
