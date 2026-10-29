# cULMblas: BLAS with CUDA, step by step

- How can BLAS operations be implemented with CUDA, and how far does one get
  by improving them one step at a time?  Starting with GEMM.
- Same approach as for the CPU part of ulmBLAS:
  - begin with something simple and correct;
  - measure it against the real thing (cuBLAS, and MAGMA where it is
    available);
  - change one thing per step, so that every step shows what it buys.
- Follows the idea of my old tutorial
  [GEMM: From Pure C to SSE Optimized Micro Kernels](https://github.com/michael-lehn/gemm-tutorial)
  (the same for one CPU core).
- Grows out of the last sessions of my lecture
  [High Performance Computing I (winter 2025/26)](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/):
  - [session 31](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session31/page01.html):
    a test and benchmark with cuBLAS;
  - [session 32](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session32/page01.html):
    a naive GEMM with CUDA;
  - the CPU side of the same story, a blocked GEMM built up step by step, is
    in the earlier sessions; see
    [ARCHITECTURE.md](../ARCHITECTURE.md#how-ulmblas-is-organized).

## Steps

- **[Step 01: naive implementation](step01/README.md)**
  - one thread per element of C
- **[Step 02: reuse a loaded element of A](step02/README.md)**
  - a thread computes 4 elements of a row of C
  - 1.55 times step 01
- **[Step 03: reuse loaded elements of A and of B](step03/README.md)**
  - a thread computes an 8 x 4 block of C
  - 1.59 times step 02 for n >= 1000, much slower for small matrices
  - why the obvious 4 x 4 block was slower than step 02
- **[Step 04: several values of l per pass](step04/README.md)**
  - a thread computes a 4 x 4 block of C, loads everything for 4 values of
    l, then computes
  - faster than step 03 for every size: 1.21 times for n >= 1000, 2.28
    times for n <= 500
  - faster than MAGMA's own kernel up to n = 600
  - why one argument of `__launch_bounds__` decides it
  - a rebuild of MAGMA's kernel, simplified step by step
- **[Step 05: panels of A and B in shared memory](step05/README.md)**
  - the threads of a block copy panels of A and B into shared memory, each
    element once, and compute from there
  - 1.19 times step 04 for n >= 1000, 1.49 times for n <= 500
  - as fast as MAGMA's own kernel: 0.98 of it for n >= 1000, faster for
    small matrices
- **[Step 06: Tensor Cores](step06/README.md)**
  - WMMA: a warp computes an 8 x 8 block of C with the FP64 Tensor Cores,
    operands through shared memory
  - 0.52 of step 05 for n >= 1000, 1.33 times for n <= 500
  - same number of Tensor Core instructions as cuBLAS, 10 times as many
    instructions in total
- **[Step 07: several 8 x 8 blocks of C per warp](step07/README.md)**
  - a warp computes 4 x 4 blocks of 8 x 8 elements of C, every fragment of
    A and B used 4 times
  - n >= 1000: 1.47 times step 05, 1.41 times MAGMA's own kernel, 0.55 of
    cuBLAS; up to 10.3 TFLOP/s
  - slower than step 05 for n <= 500
- **[Step 08: Tensor Cores with panels in shared memory](step08/README.md)**
  - the threads of a block copy panels of A and B into shared memory, each
    element once, as in step 05; the warps load their fragments from there
  - columns of the panels padded by 4 doubles: no bank conflicts
  - faster than steps 05 and 07 for every size; n >= 1000: 1.32 times step
    07, 1.88 times MAGMA's own kernel, 0.72 of cuBLAS, up to 13.7 TFLOP/s
- **[Step 09: asynchronous copies of the panels](step09/README.md)**
  - `cp.async`, 16-byte copies, two panel buffers: the next panel is on the
    way while the warps compute
  - n >= 1000: 1.05 times step 08, 0.77 of cuBLAS, up to 15.0 TFLOP/s
  - more buffers are slower: shared memory limits the blocks per SM
  - slower than step 08 for small matrices and for n = 2 mod 4
- **[Step 10: mma.sync, operands with 16-byte loads](step10/README.md)**
  - the Tensor Core operation `mma.sync` (PTX) instead of WMMA, with its
    documented assignment of operands to threads
  - values of l and rows of C assigned so that a thread loads two operands
    with one 16-byte load, without bank conflicts
  - n >= 1000: 1.03 times step 09, 0.79 of cuBLAS, up to 14.8 TFLOP/s;
    n <= 500: 1.15 times step 09
- **[Step 11: the order of the blocks](step11/README.md)**
  - why steps 08 to 10 get slower for n >= 2000: the 300 W power limit
    lowers the clock, more with more DRAM traffic
  - blocks started in groups of 8 block rows: they share their panels in
    the L2 cache; at n = 4000 2.4 instead of 10.3 GB from DRAM
  - n >= 3000: 1.09 times step 10, 13.7 TFLOP/s at n = 4000 (step 10:
    11.3); unchanged for n <= 2000
- **[Step 12: the block per warp by the size of C](step12/README.md)**
  - three kernels of step 11: 2 x 2, 4 x 2, 4 x 4 blocks of 8 x 8 per
    warp; `gemm` chooses by m*n
  - 1.33 times step 11 for n <= 500, 1.06 times for n >= 3000
  - 82% of cuBLAS (median over all sizes), up to 14.7 TFLOP/s
  - `cULMblas::gemm` uses step 12
- **[Step 13: the panels copied without checks](step13/README.md)**
  - blocks inside of C, full panels: copies without checks, pointers
    advanced from panel to panel
  - 3.2 instead of 5.9 instructions per `mma.sync` (cuBLAS 3.3), but only
    1 to 7% faster at fixed clock: the instructions were not the limit
  - 1.03 times step 12 for n >= 3000, 0.96 for n <= 500
  - 84% of cuBLAS (median over all sizes), up to 15.3 TFLOP/s
  - `cULMblas::gemm` uses step 13
- **Further steps**: in the works.  Candidates:
  - Tensor Cores busy 86 to 93% (cuBLAS 98%): waiting at the barrier and
    for the copies

## Build, test, benchmark

You need the CUDA toolkit (nvcc, cuBLAS), GNU Make and an NVIDIA GPU:

```sh
cd cuda
make                       # build/test_gemm, build/bench_gemm
make check                 # correctness tests against cuBLAS
make check-memcheck        # quick tests under compute-sanitizer
make bench                 # benchmark against cuBLAS
make bench BENCH_ARGS="--reps 20 --shapes '1000,1000,1000 4096,256,4096'"
make plot                  # the plots from the saved results (gnuplot)
make help                  # all targets and settings
```

- `CUDA_ARCH`: the GPU architecture (default `native`, the GPU of the
  machine; on a login node without a GPU e.g. `CUDA_ARCH=sm_80`).
- MAGMA is optional:
  - `make MAGMA=/path/to/magma` (an installation with `include/` and
    `lib/`; or `MAGMA_CFLAGS` and `MAGMA_LIBS`);
  - the tests and the benchmark then also run MAGMA's two DGEMMs:
    `magma_dgemm`, which in MAGMA 2.10.0 just calls `cublasDgemm` (cuBLAS
    seen through MAGMA), and `magmablas_dgemm`, MAGMA's own CUDA kernel
    (the independent comparison).

## How it is organized

- `culmblas/`: the interface `cULMblas::gemm` (double precision, column
  major, device pointers, no transposition) and the argument checks.
- `stepNN/`: one directory per step: kernel, README, benchmark plot,
  results.
- `support/`, `test/`, `bench/`: shared by all steps: the cuBLAS and MAGMA
  adapters, the list of implementations, the correctness tests, the
  benchmark.
- A new step: add it to `STEPS` in the Makefile and to the list in
  `support/support.h`; it is then tested and benchmarked like the others.
- So far: DGEMM only, without transposition, on one GPU.
