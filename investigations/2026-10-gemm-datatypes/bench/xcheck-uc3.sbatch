#!/bin/bash
#  xcheck-uc3.sbatch ARCH TARBALL TAG: the ulmBLAS test suite (make check:
#  numeric tests and the Netlib BLAS/CBLAS test programs) for the test
#  configurations in CONFIGS (environment) of an ulmBLAS source tree packed
#  in TARBALL (sha256 recorded), on a compute node of bwUniCluster 3.0.
#  Configurations whose CPU features are missing are reported as SKIP.
#  Logs: $W/ARCH/xcheck-TAG/logs/
#
#      CONFIGS="z-blis_zen4_12x4 ..." sbatch -p dev_cpu_il -J xcheck-icelake \
#          xcheck-uc3.sbatch icelake $W/up/zt.tgz z1
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1; TARBALL=$2; TAG=$3
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
CONFIGS=${CONFIGS:?CONFIGS must be set}
K=$W/$ARCH/xcheck-$TAG
rm -rf $K && mkdir -p $K/src $K/logs || exit 1
tar xzf $TARBALL -C $K/src || exit 1
cd $K/src || exit 1
module purge
module load compiler/gnu/14.2
{
    echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"
    echo "tarball $(sha256sum $TARBALL)"
    grep -m1 'model name' /proc/cpuinfo
    grep -m1 '^flags' /proc/cpuinfo | grep -o 'avx512[a-z]*' | sort -u | tr '\n' ' '; echo
    g++ --version | head -1
    echo "CONFIGS=$CONFIGS"
} > $K/logs/check-env.txt
make -j$SLURM_CPUS_PER_TASK check CXX=g++ CC=gcc FC=gfortran CONFIGS="$CONFIGS" > $K/logs/check.log 2>&1
echo "make check exit $? $(date -Iseconds)" >> $K/logs/check-env.txt
sed -n '/^Summary/,$p' $K/logs/check.log >> $K/logs/check-env.txt
