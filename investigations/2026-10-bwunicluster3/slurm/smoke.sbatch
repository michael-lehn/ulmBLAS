#!/bin/bash
#  smoke.sbatch ARCH: quick functional test of all programs built by
#  build.sbatch (one sample per size, no exclusive node; the numbers are
#  not measurements).  sbatch -p cpu_il smoke.sbatch icelake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000mb
#SBATCH --time=00:08:00
#SBATCH --output=%x-%j.out
module purge
module load compiler/gnu/14.2
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
cd $W/$1/bin || exit 1
for v in ulmblas blis-auto blis-haswell openblas mkl aocl eigen; do
    echo "== $v"
    ./bench-$v 1 1 7 7 7 100 100 100 512 512 512 && ./bench-$v 1 0 64 64 64 && ./verify-$v 768 || echo "FAILED $v"
done
./peak
