#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=build-openblas-%j.out
#
#  Builds OpenBLAS 0.3.34 with OpenMP threading and LAPACK (gfortran) on a
#  node of the target CPU type.  Usage (Ice Lake):
#
#      sbatch -p dev_cpu_il build-openblas.sbatch $W/lapack-omp/openblas
#
#  The target is detected on the build node (SKYLAKEX on Ice Lake).
set -euo pipefail
PREFIX=${1:?install prefix}
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
module load compiler/gnu/14.2
mkdir -p "$PREFIX" && cd "$PREFIX"
rm -rf OpenBLAS-0.3.34 && tar xzf $W/src/OpenBLAS-0.3.34.tar.gz
cd OpenBLAS-0.3.34
make -j 32 CC=gcc FC=gfortran USE_OPENMP=1 NUM_THREADS=128 NO_SHARED=1 \
     > "$PREFIX/make.log" 2>&1
make PREFIX="$PREFIX/install" NO_SHARED=1 install > "$PREFIX/install.log" 2>&1
{
  echo "host $(hostname)  $(date -Is)"
  gcc --version | head -1
  grep -E "^(CORE|LIBCORE|NUM_THREADS|USE_OPENMP)=" Makefile.conf || true
  cat config.h | grep -E "define (CORENAME|CHAR_CORENAME)" || true
  echo "--- objects defining dgetrf_ / dpotrf_ (OpenBLAS's own LAPACK drivers):"
  nm -A "$PREFIX/install/lib/libopenblas.a" 2>/dev/null | grep -E " T (dgetrf_|dpotrf_|dgetrf_parallel|dpotrf_L_parallel|dpotrf_U_parallel)$" || true
} > "$PREFIX/build-info.txt"
cat "$PREFIX/build-info.txt"
