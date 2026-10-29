#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=8000mb
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#  Rerun of the kernel identity probe (kernel-info.c, fixed) of the native
#  BLIS and AOCL builds of xrun-8028076 on a node of the same partition.
set -u
module purge; module load compiler/gnu/14.2
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench; D=$W/up/xbench; R=$W/icelake/refs
AOCL=/software/bwhpc/common/lib/aocl/5.0.0/aocc
O=$W/icelake/zprobe-$SLURM_JOB_ID; mkdir -p $O && cd $O || exit 1
PIN="numactl --physcpubind=3 --membind=0"
{
    echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"; lscpu | grep 'Model name'
    sha256sum $D/kernel-info.c
    for b in blis-auto blis-haswell; do
        echo "== $b"
        gcc -O2 -DPREC=z -DBLIS -I$R/$b/include/blis $D/kernel-info.c $R/$b/lib/libblis.a -lpthread -lm -o info-$b && $PIN ./info-$b
    done
    echo "== aocl"
    gcc -O2 -DPREC=z -DBLIS -DBLIS_0X -I$AOCL/include $D/kernel-info.c $AOCL/lib/libblis.a -lpthread -lm -o info-aocl && $PIN ./info-aocl
} > kernel-identity.txt 2>&1
