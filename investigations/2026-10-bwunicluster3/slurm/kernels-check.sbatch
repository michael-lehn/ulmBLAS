#!/bin/bash
#  kernels-check.sbatch ARCH REV: the ulmBLAS test suite (make check: numeric
#  tests and the Netlib BLAS/CBLAS test programs) for the test
#  configurations of all BLIS- and OpenBLAS-derived DGEMM micro kernels
#  (blis-NAME, openblas-NAME with the default block sizes, ...-small with
#  small block sizes) and for the
#  configuration testcase06 (ulmBLAS's AVX/FMA kernel), for ulmBLAS at git
#  revision REV (exported to $W/ARCH/check-REV/src).  Configurations whose
#  CPU features are missing are reported as SKIP.  Logs:
#  $W/ARCH/check-REV/logs/
#
#      sbatch -p cpu_il -J kcheck-icelake kernels-check.sbatch icelake REV
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60000mb
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1; REV=$2
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
FULL=$(git -C $W/src/ulmBLAS rev-parse $REV) || exit 1
K=$W/$ARCH/check-$REV
rm -rf $K && mkdir -p $K/src $K/logs || exit 1
git -C $W/src/ulmBLAS archive $FULL | tar xf - -C $K/src || exit 1
cd $K/src || exit 1
module purge
module load compiler/gnu/14.2
CONFIGS="testcase06"
for n in haswell_6x8 haswell_8x6 haswell_12x4 haswell_4x12 sandybridge_8x4 sandybridge_int_8x4 penryn_4x4 skx_16x14 skx_16x12_l2 zen4_32x6 zen4_8x24 zen5_8x24; do
    CONFIGS="$CONFIGS blis-$n blis-$n-small"
done
for n in skylakex_16x12; do
    CONFIGS="$CONFIGS openblas-$n openblas-$n-small"
done
echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds) ulmBLAS $FULL" > $K/logs/check-env.txt
grep -m1 'model name' /proc/cpuinfo >> $K/logs/check-env.txt
echo "CONFIGS=$CONFIGS" >> $K/logs/check-env.txt
make -j$SLURM_CPUS_PER_TASK check CXX=g++ CC=gcc FC=gfortran CONFIGS="$CONFIGS" > $K/logs/check.log 2>&1
echo "make check exit $? $(date -Iseconds)" >> $K/logs/check-env.txt
sed -n '/^Summary/,$p' $K/logs/check.log >> $K/logs/check-env.txt
