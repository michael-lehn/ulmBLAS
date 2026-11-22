#!/bin/sh
# Builds all examples with GCC and clang and stores their outputs in out/.
# macOS with Homebrew: g++-16 and the clang of Homebrew's llvm.
set -e
cd "$(dirname "$0")"
GXX=${GXX:-g++-16}
CLANGXX=${CLANGXX:-clang++}
FLAGS="-std=c++17 -O3 -mcpu=native -Wall -Wextra"
mkdir -p out bin
export OMP_NUM_THREADS=4

gcc_simd()   { $GXX     $FLAGS -fopenmp-simd "$1" -o "bin/$2.gcc"; }
clang_simd() { $CLANGXX $FLAGS -fopenmp-simd "$1" -o "bin/$2.clang"; }
gcc_omp()    { $GXX     $FLAGS -fopenmp "$1" -o "bin/$2.gcc"; }
clang_omp()  { $CLANGXX $FLAGS -fopenmp "$1" -o "bin/$2.clang"; }

# Part I: vectorization (no OpenMP runtime)
$GXX -std=c++17 -O2 -fopenmp-simd -fopt-info-vec-all -c simd/axpy.cc \
    -o /dev/null 2>&1 | grep -E "^simd/axpy.cc:[0-9]+:[0-9]+: (optimized|missed): (loop vectorized|.*versioning|.*alias check|couldn't vectorize loop|loop versioned)" \
    | sed 's/  */ /g' > out/axpy-gcc.txt
$CLANGXX -std=c++17 -O2 -fopenmp-simd -Rpass=loop-vectorize -c simd/axpy.cc \
    -o /dev/null 2>&1 | grep "remark" > out/axpy-clang.txt
for p in sum lanes scal iamax clauses; do
    gcc_simd simd/$p.cc $p; clang_simd simd/$p.cc $p
    bin/$p.gcc > out/$p-gcc.txt; bin/$p.clang > out/$p-clang.txt
done

# Part II: threads
for p in threads/hello threads/race threads/falsesharing threads/nested \
         first/shares first/cholesky1 atomics/cas atomics/barrier \
         forkjoin/constructs forkjoin/gemv forkjoin/pool forkjoin/gemm \
         tasks/basics tasks/depend tasks/cholesky tasks/team; do
    n=$(basename $p)
    gcc_omp $p.cc $n; clang_omp $p.cc $n
    bin/$n.gcc > out/$n-gcc.txt; bin/$n.clang > out/$n-clang.txt
done
$GXX $FLAGS atomics/mp.cc -o bin/mp.gcc
$CLANGXX $FLAGS atomics/mp.cc -o bin/mp.clang
bin/mp.gcc > out/mp-gcc.txt; bin/mp.clang > out/mp-clang.txt
$GXX $FLAGS atomics/sb.cc -o bin/sb.gcc
$CLANGXX $FLAGS atomics/sb.cc -o bin/sb.clang
bin/sb.gcc > out/sb-gcc.txt; bin/sb.clang > out/sb-clang.txt
# out/sb-heim-gcc.txt: the same program on an Intel Core i5-12500 (x86-64),
# GCC 12 with -march=native, run by hand

OMP_WAIT_POLICY=active bin/barrier.clang > out/barrier-clang-active.txt
OMP_WAIT_POLICY=active bin/pool.clang > out/pool-clang-active.txt
OMP_MAX_TASK_PRIORITY=3 bin/cholesky.gcc > out/cholesky-gcc-prio.txt
OMP_MAX_TASK_PRIORITY=3 bin/cholesky.clang > out/cholesky-clang-prio.txt
echo done
