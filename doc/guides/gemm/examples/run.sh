#!/bin/sh
# Builds the examples and stores the outputs shown in the guide in out/.
#   ULMBLAS: a clone of ulmBLAS at tag v0.4.2 (default: ~/git/ulmBLAS)
# The measurements are made on one thread (ULMBLAS_NUM_THREADS=1) and need
# an otherwise idle machine.  macOS with Homebrew: g++-16 and the clang of
# Homebrew's llvm.  On x86-64 with AVX2 the assembly kernel of ulmBLAS is
# compiled along.
cd "$(dirname "$0")"
ULMBLAS=${ULMBLAS:-$HOME/git/ulmBLAS}
GXX=${GXX:-g++-16}
CLANGXX=${CLANGXX:-clang++}
case $(uname -m) in
x86_64)
    ARCH=-march=native
    ASM=$ULMBLAS/ulmblas/level3/ukernel/openblas/asm/haswell_4x8.S ;;
*)
    ARCH=-mcpu=native
    ASM= ;;
esac
FLAGS="-std=c++11 -O3 $ARCH -fopenmp-simd -Wall -Wextra -isystem $ULMBLAS"
mkdir -p out bin
export ULMBLAS_NUM_THREADS=1

# Chapter 2: caches
$GXX $FLAGS caches/chase.cc -o bin/chase && bin/chase > out/chase.txt
$GXX $FLAGS caches/assoc.cc -o bin/assoc && bin/assoc > out/assoc.txt
sysctl hw.cachelinesize hw.perflevel0.l1dcachesize \
    hw.perflevel0.l2cachesize hw.perflevel0.cpusperl2 hw.pagesize \
    > out/sysctl.txt 2>/dev/null

# Chapter 3: loop orders
$GXX $FLAGS naive/loops.cc -o bin/loops && bin/loops > out/loops.txt

# Chapters 5-8: the blocked gemm
for c in "$GXX gcc" "$CLANGXX clang"; do
    set -- $c
    $1 $FLAGS blocked/steps.cc $ASM -o bin/steps.$2 &&
	bin/steps.$2 > out/steps-$2.txt
done
$GXX $FLAGS blocked/blocksizes.cc -o bin/blocksizes &&
    bin/blocksizes > out/blocksizes.txt
$GXX $FLAGS blocked/packing.cc -o bin/packing && bin/packing > out/packing.txt

# Chapters 11-13: the other level-3 functions (GCC: on the M5 the NEON
# kernels, the kernels of the examples)
for p in symm syrk trsm all; do
    $GXX $FLAGS level3/$p.cc $ASM -o bin/$p && bin/$p > out/$p.txt
done

# Chapters 14 and 15: small matrices, the run-time selection
$GXX $FLAGS small/small.cc $ASM -o bin/small && bin/small > out/small.txt
$GXX $FLAGS selection/select.cc $ASM -o bin/select && bin/select > out/select.txt
ULMBLAS_PLATFORM=generic bin/select > out/select-generic.txt
