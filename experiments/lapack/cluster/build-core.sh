#!/bin/bash
#  build-core.sh SRC: builds the benchmarks of the ulmBLAS templates
#  (bench-core-{ulmblas,openblas,mkl}) in SRC/experiments/lapack, with the
#  fastest micro kernels of the GEMM campaigns on Ice Lake for every type
#  (PERFORMANCE.md): OpenBLAS SKYLAKEX for s, d, c, AOCL's zen4 12x4 for z
set -euo pipefail
source "$(dirname "$0")/env.sh"
module load numlib/mkl/2025.0 >/dev/null 2>&1
cd "$1/experiments/lapack"
ARCH="-mavx -mfma -mavx2 -mavx512f -mavx512dq -mavx512vl"
ARCH="$ARCH -DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4"
ARCH="$ARCH -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12"
ARCH="$ARCH -DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2"
ARCH="$ARCH -DULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4"
make -j 16 CXX=g++ "ARCHFLAGS=$ARCH" OMPFLAGS=-fopenmp OMPLIBS=-fopenmp \
     "REFLIBS=$OB/lib/libopenblas.a -lgfortran -lm" \
     build/bench-core-ulmblas build/bench-core-openblas build/bench-core-mkl \
     > build-core.log 2>&1 || { tail -30 build-core.log; exit 1; }
grep -E "warning|error" build-core.log || true
echo "ARCHFLAGS: $ARCH"
