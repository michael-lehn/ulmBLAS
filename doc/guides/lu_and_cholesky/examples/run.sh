#!/bin/sh
# Builds all examples with GCC and clang against the headers of ulmBLAS and
# stores the outputs shown in the guide in out/.
#   ULMBLAS: a clone of ulmBLAS at tag v0.4.2 (default: ~/git/ulmBLAS)
# The headers of ulmBLAS are included with -isystem, so that -Wextra
# reports only warnings about the examples themselves.
# macOS with Homebrew: g++-16 and the clang of Homebrew's llvm.
cd "$(dirname "$0")"
ULMBLAS=${ULMBLAS:-$HOME/git/ulmBLAS}
GXX=${GXX:-g++-16}
CLANGXX=${CLANGXX:-clang++}
FLAGS="-std=c++11 -O3 -mcpu=native -fopenmp-simd -Wall -Wextra -isystem $ULMBLAS"
mkdir -p out bin

# run <file> <name>: build with both compilers and run
run() {
    $GXX $FLAGS $1 -o bin/$2.gcc && bin/$2.gcc > out/$2-gcc.txt
    $CLANGXX $FLAGS $1 -o bin/$2.clang && bin/$2.clang > out/$2-clang.txt
}

# Chapter 2: basics
run basics/pivot.cc pivot
run basics/safemin.cc safemin
run basics/nan.cc nan

# Chapters 3 to 5: the variants, on one thread
(
    export ULMBLAS_NUM_THREADS=1
    run variants/variants.cc variants
)

# Chapter 6: blocked and recursive (GCC), on one thread and on all
$GXX $FLAGS blocked/getrf.cc -o bin/getrf
ULMBLAS_NUM_THREADS=1 bin/getrf > out/getrf-1.txt
bin/getrf > out/getrf-all.txt
$GXX $FLAGS blocked/potrf.cc -o bin/potrf
ULMBLAS_NUM_THREADS=1 bin/potrf > out/potrf-1.txt
bin/potrf > out/potrf-all.txt

# Chapter 7: the base cases
run variants/potf2.cc potf2

# Chapter 13: solving
run solve/inverse.cc inverse
