#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=test-%j.out
#  test.sbatch SRC: build and run the correctness tests on an Ice Lake node
set -euo pipefail
SRC=$1
source $SRC/lapack/cluster/env.sh
bash $SRC/lapack/cluster/build.sh $SRC
cd $SRC/lapack
echo "host $(hostname)  $(date -Is)  $(lscpu | grep 'Model name')"
for t in 1 32; do
    echo "== OMP_NUM_THREADS=$t (configurations with up to 32 threads)"
    OMP_NUM_THREADS=$t TEST_MAX_THREADS=32 ./build/test | tail -2
done
echo "== kernel bounds"
make -s "${MAKEVARS[@]}" check-bounds
echo "== lost updates next to the output tile"
make -s "${MAKEVARS[@]}" check-neighbor
echo "== dependency checker (full test suite)"
make -s "${MAKEVARS[@]}" check-deps 2>&1 | tail -2
echo "== bitwise determinism (120 s)"
make -s "${MAKEVARS[@]}" DETERMINISM_SECONDS=120 check-determinism | tail -3
