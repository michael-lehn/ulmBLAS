#!/bin/bash
#  full-build.sbatch ARCH REV: the ulmBLAS programs of the full DGEMM
#  campaign (full-bench.sbatch), all from git revision REV, with
#  kernels-build.sbatch (which must be in the same directory):
#
#    tag f1:  ulmBLAS's own AVX/FMA kernel, BLIS skx 16x14, AOCL zen4 32x6
#    tag it3: OpenBLAS SKYLAKEX 16x12, final version (tuning iteration 3:
#             one kernel call per panel of B, B packed interleaved)
#    tag it2: the same with -DULMBLAS_DGEMM_NO_INTERLEAVE (iteration 2)
#    tag it1: the same with -DULMBLAS_DGEMM_NO_PANEL
#             -DULMBLAS_DGEMM_NO_INTERLEAVE (iteration 1: one call per tile)
#
#      sbatch -p dev_cpu_il -J fbuild-icelake full-build.sbatch icelake REV
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1; REV=$2
D=${SLURM_SUBMIT_DIR:-.}/..
KB=$D/kernels-build.sbatch
[ -f $KB ] || KB=$(dirname $0)/kernels-build.sbatch
st=0
KERNELS="avx blis_skx_16x14 blis_zen4_32x6" bash $KB $ARCH $REV f1 || st=1
KERNELS=openblas_skylakex_16x12 bash $KB $ARCH $REV it3 || st=1
KERNELS=openblas_skylakex_16x12 EXTRA="-DULMBLAS_DGEMM_NO_INTERLEAVE" \
    bash $KB $ARCH $REV it2 || st=1
KERNELS=openblas_skylakex_16x12 EXTRA="-DULMBLAS_DGEMM_NO_PANEL -DULMBLAS_DGEMM_NO_INTERLEAVE" \
    bash $KB $ARCH $REV it1 || st=1
exit $st
