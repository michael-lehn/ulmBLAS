#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=core-test-%j.out
#
#  core-test.sbatch SRC OUT: tests of ?getrf and ?potrf of ulmBLAS on Ice
#  Lake (GCC 14.2): the C++ tests with OpenMP (two test configurations),
#  the factorization tests with 1, 8 and 32 threads and 30 times with 32
#  threads, and the LAPACK 3.12.1 LIN test programs (paths GE, PO).
#  Output in OUT.
#
set -uo pipefail
SRC=$1; OUT=$2
source $SRC/experiments/lapack/cluster/env.sh
mkdir -p $OUT
cd $SRC
make -C test check-numeric OPENMP=1 CXX=g++ CONFIGS="testcase01 testcase04" -j 16 \
    > $OUT/check-numeric.log 2>&1
grep -A3 '^Summary' $OUT/check-numeric.log
T=test/build/testcase04/numeric/numeric_tests
F="T0410 T0411 T0412 T0413 T0414 T0415 T0416 T0417"
for t in 1 8 32; do
    echo "threads $t: $(OMP_NUM_THREADS=$t "${NUMA[@]}" $T $F | grep -c '^PASS') of 8 PASS"
done
fail=0
for i in $(seq 1 30); do
    OMP_NUM_THREADS=32 "${NUMA[@]}" $T $F | grep -q FAIL && fail=$((fail+1))
done
echo "30 runs with 32 threads: $fail failed"
make check-reference-lapack-lin CXX=g++ FC=gfortran > $OUT/lin.log 2>&1
tail -6 $OUT/lin.log
