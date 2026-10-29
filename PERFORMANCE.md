# ulmBLAS performance

This page collects all my benchmarks of ulmBLAS: the machines, the
operations I measured and the results, with links to the benchmark
campaigns in [benchmark_results/](benchmark_results/README.md) that
document hardware, software, methodology and the raw data.  The
experiments with the SME2 backend on the Apple M5 are in
[M5_EXPERIMENTS.md](M5_EXPERIMENTS.md).

**We are competitive, but not yet where we want to be.**

With architecture-specific micro kernels, ulmBLAS already comes close to
highly optimized libraries such as OpenBLAS and Intel oneMKL.  On an Intel
Xeon (Ice Lake), for example, single-threaded DGEMM in ulmBLAS reaches
about 97% of OpenBLAS's performance for large matrices (n ≥ 1000) and
about 99% of oneMKL's, using OpenBLAS's own micro kernel inside ulmBLAS's
GEMM implementation
([campaign 023](benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md)).

We find that encouraging, given that we keep the implementation simple
and readable.  But matching established libraries on large matrices is
only part of the challenge.  Small matrices are still a weakness (for
n = 50 the same build reaches only about 60% of OpenBLAS), and even the
last few percent for large matrices deserve a closer look.

**We are not done yet.**

## Machines

These are the machines I have access to.  If you run
[`make bench-gemm`](ARCHITECTURE.md#how-to-run-your-own-benchmarks) on another
one, I'd be glad to hear about the results: they help to see where
ulmBLAS stands.

| Machine | CPU | Architecture, instruction sets | Operating system | Campaigns |
|---|---|---|---|---|
| MacBook Air | Apple M5 (4 performance, 6 efficiency cores) | AArch64: NEON, SME2 (SVL 512 bit) | macOS 26.6 | [002](benchmark_results/002-2026-10-24-macbook-m5-neon/README.md), [003](benchmark_results/003-2026-10-25-macbook-m5-profile/README.md), [004](benchmark_results/004-2026-10-25-macbook-m5-sme/README.md), [005](benchmark_results/005-2026-10-25-macbook-m5-sme-blocksizes/README.md), [006](benchmark_results/006-2026-10-25-macbook-m5-sme-square/README.md), [007](benchmark_results/007-2026-10-26-macbook-m5-sme-small/README.md), [022](benchmark_results/022-2026-10-27-macbook-m5-sme-blocking/README.md), [029](benchmark_results/029-2026-10-28-macbook-m5-zgemm/README.md) |
| heim | Intel Core i5-12500 (Alder Lake, 6 cores) | x86-64: SSE3, AVX, AVX2/FMA | Debian 12 | [001](benchmark_results/001-2026-10-24-heim-avx-fma/README.md), [008](benchmark_results/008-2026-10-26-heim-avx-fma-memory/README.md), [009](benchmark_results/009-2026-10-26-heim-eigen/README.md), [011](benchmark_results/011-2026-10-26-heim-blis-kernels/README.md), [013](benchmark_results/013-2026-10-27-heim-blis-kernels-mc/README.md), [017](benchmark_results/017-2026-10-27-heim-blis-kernels-old/README.md), [024](benchmark_results/024-2026-10-27-heim-openblas-haswell-kernel/README.md), [026](benchmark_results/026-2026-10-27-heim-small-dgemm/README.md), [027](benchmark_results/027-2026-10-28-heim-zgemm/README.md) |
| bwUniCluster 3.0, partition `cpu_il` | Intel Xeon Platinum 8358 (Ice Lake-SP, 2 x 32 cores), one exclusive node | x86-64: up to AVX-512 | RHEL 9.6 | [010](benchmark_results/010-2026-10-26-uc3-icelake-dgemm/README.md), [012](benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md), [014](benchmark_results/014-2026-10-27-uc3-icelake-openblas-kernel/README.md), [016](benchmark_results/016-2026-10-27-uc3-icelake-openblas-kernel-prof/README.md), [019](benchmark_results/019-2026-10-27-uc3-icelake-kernels-k4/README.md), [020](benchmark_results/020-2026-10-27-uc3-icelake-openblas-kernel-panel/README.md), [021](benchmark_results/021-2026-10-27-uc3-icelake-openblas-kernel-interleave/README.md), [023](benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md), [025](benchmark_results/025-2026-10-27-uc3-icelake-small-dgemm-profile/README.md), [028](benchmark_results/028-2026-10-28-uc3-icelake-zgemm/README.md), [030](benchmark_results/030-2026-10-28-uc3-icelake-sgemm/README.md), [031](benchmark_results/031-2026-10-29-uc3-icelake-cgemm/README.md) |
| bwUniCluster 3.0, partition `cpu` | AMD EPYC 9454 (Zen 4), one socket of a shared node (preliminary) | x86-64: up to AVX-512 | RHEL 9.6 | [015](benchmark_results/015-2026-10-27-uc3-zen4-kernels-socket/README.md), [018](benchmark_results/018-2026-10-27-uc3-zen4-kernels-k4-socket/README.md) |

All measurements are single-threaded (one core), with medians of
several interleaved rounds; the libraries compared are named in each
section.

## Operations

- [General matrix product (GEMM)](#general-matrix-product-gemm):
  [DGEMM](#dgemm-double), [ZGEMM](#zgemm-complex-double),
  [SGEMM](#sgemm-float), [CGEMM](#cgemm-complex-float)
- [Symmetric matrices](#symmetric-matrices-dsymm-dsyrk): dsymm, dsyrk
- [Triangular matrices](#triangular-matrices-dtrmm-dtrsm): dtrmm, dtrsm

All level 3 routines of ulmBLAS are built on the GEMM micro kernel of
their data type, so the GEMM results carry over to them; the sections
below show where they differ.

## General matrix product (GEMM)

C ← alpha A B + beta C.  The results are grouped by data type and
ordered newest first within a type.

### DGEMM (double)

#### x86-64: AVX-512 on Intel Xeon (Ice Lake), single-threaded DGEMM

**Single-threaded DGEMM on one core** of an Intel Xeon Platinum 8358
(Ice Lake-SP, AVX-512; bwUniCluster 3.0, one exclusive node), m = n = k =
50, 100, …, 4000, double precision, column major, alpha = beta = 1.
ulmBLAS (71f1ce1, GCC 14.2) with four different micro kernels, against
OpenBLAS 0.3.34 (SKYLAKEX), oneMKL 2025.0.1 (sequential), BLIS 2.0 (skx),
AOCL-BLAS 5.0 and Eigen 5.0.1 (`C.noalias() += A*B`).  Medians of three
interleaved rounds; every variant is its own curve:

![Single-threaded DGEMM n = 50 ... 4000 on an Intel Xeon Platinum 8358: ulmBLAS with four micro kernels, OpenBLAS, oneMKL, Eigen, BLIS and AOCL-BLAS](benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/plots/dgemm_full.svg)

The best ulmBLAS build uses **OpenBLAS's SKYLAKEX micro kernel**
(16 x 12, AVX-512), vendored into ulmBLAS
([ulmblas/level3/ukernel/openblas](ulmblas/level3/ukernel/openblas/README.md))
and called from ulmBLAS's own GEMM algorithm, packing and blocking.  The
other builds use AOCL-BLAS's `zen4` 32 x 6 and BLIS's `skx` 16 x 14 kernel
and ulmBLAS's own AVX2/FMA 6 x 8 kernel.

| n = m = k | ulmBLAS + OpenBLAS kernel | OpenBLAS | oneMKL | Eigen | BLIS | ulmBLAS / OpenBLAS | ulmBLAS / oneMKL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 35.0 | 59.3 | 58.4 | 50.7 | 26.9 | 0.59 | 0.60 |
| 100 | 59.7 | 75.6 | 74.6 | 72.9 | 52.4 | 0.79 | 0.80 |
| 200 | 77.0 | 83.3 | 82.2 | 87.1 | 66.9 | 0.92 | 0.94 |
| 500 | 86.9 | 90.8 | 88.8 | 89.8 | 75.8 | 0.96 | 0.98 |
| 1000 | 92.0 | 95.2 | 94.9 | 95.9 | 79.4 | 0.97 | 0.97 |
| 2000 | 94.7 | 97.5 | 94.6 | 96.4 | 81.2 | 0.97 | 1.00 |
| 4000 | 96.7 | 99.4 | 98.2 | 97.4 | 81.7 | 0.97 | 0.98 |

(GFLOP/s, medians of 3 rounds; ratios: medians of the per-round ratios.
The 512-bit FMA peak of the core, measured with a microbenchmark, is 105.2
GFLOP/s.)

- **Large matrices** (n ≥ 1000, 61 sizes): ulmBLAS with OpenBLAS's
  kernel reaches 92.0 to 96.7 GFLOP/s (up to 92% of the FMA peak): 0.971
  of OpenBLAS (0.958 … 0.979), 0.989 of oneMKL (0.969 … 1.012), 0.995 of
  Eigen, 1.17 times BLIS.  The AOCL `zen4` kernel inside ulmBLAS reaches
  0.960 of OpenBLAS; with BLIS's `skx` kernel ulmBLAS is as fast as BLIS
  (1.006), about 0.84 of OpenBLAS.  ulmBLAS's own kernel uses 256-bit
  registers only and reaches half of OpenBLAS (48 to 49 GFLOP/s).
- **OpenBLAS is the fastest library** at 73 of the 80 sizes, Eigen at 7;
  ulmBLAS is not the fastest at any size, and it is slower than OpenBLAS
  at all 80 sizes in all three rounds.
- **Small matrices are the weak range**: for n ≤ 200 every ulmBLAS build
  is clearly behind OpenBLAS, oneMKL and Eigen (n = 50: 0.59 of OpenBLAS;
  n = 100: 0.79).  ulmBLAS has no special path for small matrices on
  x86-64.
- **Tuning**: two changes to the GEMM algorithm for OpenBLAS's kernel
  (one call per panel of B, B packed interleaved with the first block of
  A) raised the ratio to OpenBLAS for n ≥ 1000 from 0.943 to 0.971 in this
  campaign ([iterations](benchmark_results/021-2026-10-27-uc3-icelake-openblas-kernel-interleave/README.md#the-three-tuning-iterations)).
- **Correctness and repeatability**: 2640 measurements, no wrong result
  (checked against a naive product up to n = 550 and by 2000 random
  elements per size against long double sums for n ≥ 600); the medians of
  the shared variants reproduce three earlier campaigns on other nodes
  within 0.4%.

Only one core was used: the libraries differ much more in their
multi-threaded behavior, which was not measured.  Only square matrices,
on a grid of step 50, with hot caches (each sample repeats the call on
the same operands).  The
[benchmark campaign](benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md)
documents hardware, library builds and the kernels each library selects
on this CPU, the methodology, the differences of Eigen's API, all
results per size range, the crossovers, the raw data and the figures for
n ≤ 500 and relative to OpenBLAS and oneMKL.

#### AArch64: SME2 (Apple M5, experimental)

Experiments with this backend (block sizes, small matrices):
[M5_EXPERIMENTS.md](M5_EXPERIMENTS.md).

The Apple M5 has a matrix unit programmed with the Scalable Matrix
Extension (SME2): in streaming mode the outer product instruction FMOPA
accumulates into the ZA array, eight 8 x 8 tiles of doubles.  The
experimental SME2 backend uses it for DGEMM and thereby for all double
precision level 3 routines (and, with kernels of their own, for the other
three types).  For DGEMM it consists of a 32 x 16 micro kernel in C++
with the ACLE SME intrinsics (derived from the kernel of
[BLIS pull request #921](https://github.com/flame/blis/pull/921), see
[License](README.md#license-provenance-and-acknowledgments)) and routines, written
for ulmBLAS, that pack single panels in streaming mode; the GEMM algorithm
and the macro kernel are those of all other kernels.

**Requirements**: a CPU with SME2 and SME_F64F64 and a streaming vector
length of 512 bits (tested only on an Apple M5), clang (tested with Apple
clang 21; LLVM clang 23 compiles it; GCC 16 compiles the kernel, but its
Darwin runtime lacks `__arm_get_current_vg`) and `ARCHFLAGS=-mcpu=apple-m4` (or
other flags that enable SME2).  Without such flags the NEON kernel is used
as before; `CPPFLAGS=-DULMBLAS_NO_SME` disables the backend.  There is no
runtime detection: a library built with SME2 needs a CPU with SME2.

DGEMM on one performance core of an Apple M5 (MacBook Air), single
threaded, compared with Apple Accelerate (`VECLIB_MAXIMUM_THREADS=1`), the
NEON kernel of ulmBLAS and BLIS pull request #921 (medians of three
interleaved rounds, GFLOP/s):

![DGEMM performance versus matrix size on an Apple M5 with SME2](benchmark_results/004-2026-10-25-macbook-m5-sme/plots/dgemm_square.svg)

| m x n x k | ulmBLAS SME2 | Apple Accelerate | ulmBLAS NEON | BLIS PR #921 (SME2) |
|---|---|---|---|---|
| 32 x 32 x 32 | 137 | 193 | 38 | 32 |
| 128 x 128 x 128 | 342 | 432 | 56 | 238 |
| 512 x 512 x 512 | 467 | 470 | 64 | 452 |
| 1024 x 1024 x 1024 | 457 | 425 | 66 | 436 |
| 4096 x 4096 x 4096 | 477 | 428 | 66 | 473 |
| 2000 x 2000 x 32 | 177 | 130 | 55 | 143 |
| 4096 x 96 x 4096 | 365 | 258 | 56 | 352 |
| 64 x 2048 x 2048 | 286 | 320 | 57 | 247 |
| dtrsm, n = 2048 | 176 | 284 | 61 | 153 |

The FMOPA peak of the core, measured with a microbenchmark of the
instruction throughput, is 542 GFLOP/s; the micro kernel alone reaches
507 to 518.  For large matrices and most of the measured shapes with small
k or small n the SME2 backend is faster than Accelerate (1.08 to 1.13
times for n = m = k ≥ 1024), but **Accelerate is faster** for square
matrices up to n = 768 (about equal at n = 512), for small m (m ≤ 96), for
1024 x 1024 x 32 and for dtrsm, and for very small matrices (n = 8) the NEON
kernel is faster than SME2 (entering streaming mode has a fixed cost).
The numbers come from one MacBook Air; absolute values varied by a few
percent between sessions, so comparisons are made only within interleaved
runs.  The [benchmark campaign](benchmark_results/004-2026-10-25-macbook-m5-sme/README.md)
documents hardware, software, configurations, methodology, the raw data
and the correctness tests.

**A fixed configuration.**  Like every kernel of ulmBLAS, the backend uses
one set of block sizes, chosen at compile time: MC, KC, NC = 384, 1024,
1024 for double.  Different block sizes are best for different shapes:
the block sizes of BLIS pull request #921 (608, 2240, 3072) are 3 to 5%
faster for m = n = k = 2048 to 4096, the chosen ones 19 to 43% faster for
m = 16 to 64; no single configuration was best for all shapes measured.
A [study of 12 fixed configurations](benchmark_results/005-2026-10-25-macbook-m5-sme-blocksizes/README.md)
shows where their performance curves cross for square matrices and for
small m, n and k, and which differences are larger than the noise.

**Roadmap.**  A more flexible architecture is under investigation that
selects the GEMM configuration depending on the problem dimensions —
potentially block sizes, packing routines, micro kernels and loop
orders.  This
continues an old idea: ATLAS (Automatically Tuned Linear Algebra Software,
R. Clint Whaley and Jack Dongarra) searched for the best parameters and
kernels of a machine empirically when it was installed.  ulmBLAS does not
implement automatic tuning; so far the configurations are chosen by hand
from documented benchmark campaigns.

#### x86-64: AVX/FMA

DGEMM on one core of an Intel Core i5-12500 (Alder Lake, AVX2/FMA), single
threaded, compared with BLIS 2.0 and OpenBLAS 0.3.34 built from source for the
same machine:

![DGEMM performance versus matrix size](benchmark_results/001-2026-10-24-heim-avx-fma/plots/dgemm_square.svg)

| n = m = k ≥ 256 | AVX/FMA kernel relative to |
|---|---|
| ulmBLAS SSE kernel | 2.76 – 2.89 |
| BLIS 2.0 (same assembly kernel) | 0.99 – 1.02 |
| OpenBLAS 0.3.34 | 0.96 – 1.00 |

The AVX/FMA kernel reaches up to 59.7 GFLOP/s (OpenBLAS: 61.2 GFLOP/s); the
other level 3 routines reach 2.3 to 2.9 times the performance of the SSE
kernel.  For small matrices (n ≤ 128) BLIS and OpenBLAS are partly faster.

The figures show ulmBLAS e195bea.  Measured again with febb852, DGEMM is
1.12 to 1.81 times faster for n ≤ 16 (0.061 instead of 0.110 µs at n = 2),
because the packing buffers are now kept in a per-thread cache; from n = 40
on, DGEMM is unchanged (within ±1.5%), dsyrk is 3 to 4% faster for n ≥ 2048,
and dtrsm is 2.4% slower for n = 512 (unchanged for larger n)
([follow-up campaign](benchmark_results/008-2026-10-26-heim-avx-fma-memory/README.md)).
OpenBLAS remains faster for n ≤ 20.

![Speedup of the AVX/FMA micro kernel over the SSE micro kernel](benchmark_results/001-2026-10-24-heim-avx-fma/plots/dgemm_speedup.svg)

These numbers hold for this CPU and the documented configurations only.  The
[benchmark campaign](benchmark_results/001-2026-10-24-heim-avx-fma/README.md)
documents hardware, compilers, library versions and build options, source
revisions and methodology, and contains the raw measurements and further
figures (thin matrices, syrk/symm/trmm/trsm, choice of block sizes).

##### Comparison with Eigen

Eigen 5.0.1 (`C.noalias() += A*B` with column-major
`Eigen::Matrix<double, Dynamic, Dynamic, ColMajor>`, `g++ -O3 -march=native`,
Eigen's own AVX2/FMA kernels, no external BLAS, one thread), measured in the
same rounds as ulmBLAS febb852, BLIS 2.0 and OpenBLAS 0.3.34:

![DGEMM performance versus matrix size, with Eigen](benchmark_results/009-2026-10-26-heim-eigen/plots/dgemm_square.svg)

![DGEMM performance for n = 1 … 64, with Eigen](benchmark_results/009-2026-10-26-heim-eigen/plots/sweep_small.svg)

| Sizes | Fastest | Eigen |
|---|---|---|
| n ≤ 41 | Eigen (except n = 7: OpenBLAS) | 0.012 µs at n = 2 (OpenBLAS 0.034, ulmBLAS 0.061); 1.04 – 1.34 × OpenBLAS for n = 8 … 41 |
| n = 42 … 128 | BLIS or Eigen for n ≤ 64, BLIS for n = 80 … 128 | 0.88 – 0.97 × BLIS where BLIS is clearly faster |
| n = 160 … 768 | Eigen and OpenBLAS (within the noise), except n = 192 (and n = 384 in some runs) | 1.01 – 1.05 × ulmBLAS; 0.78 × at n = 192 |
| n ≥ 1024 | OpenBLAS | 52.7 – 55.6 GFLOP/s: 0.86 – 0.92 × OpenBLAS, 0.89 – 0.95 × ulmBLAS |

For n ≤ 6 Eigen uses an inlined coefficient-based product instead of its
matrix-product kernel; the BLAS libraries are called through `cblas_dgemm`.
The [Eigen campaign](benchmark_results/009-2026-10-26-heim-eigen/README.md)
documents configuration, methodology, correctness checks and all data;
ulmBLAS, BLIS and OpenBLAS reproduce the follow-up campaign within 0.97 to
1.01.

#### AArch64: NEON (Apple silicon)

DGEMM on one performance core of an Apple M5 (MacBook Air), single
threaded, compared with BLIS 2.0 (configuration firestorm) and OpenBLAS
0.3.34 built from source for the same machine:

![DGEMM performance versus matrix size on an Apple M5](benchmark_results/002-2026-10-24-macbook-m5-neon/plots/dgemm_square.svg)

| n = m = k ≥ 256 | NEON kernel relative to |
|---|---|
| ulmBLAS reference kernel (used on ARM before) | 1.96 – 2.06 |
| BLIS 2.0 (same assembly kernel) | 0.986 – 0.994 |
| OpenBLAS 0.3.34 | 1.07 – 1.14 |

The NEON kernel reaches up to 66.2 GFLOP/s, 94 to 96% of the measured FMA
peak of the core (68.9 GFLOP/s) for n ≥ 1024; the other level 3 routines
reach 1.8 to 2.1 times the performance of the reference kernel and 0.96 to
1.02 of BLIS.  For thin k (k = 32, 64) ulmBLAS is faster than BLIS and
OpenBLAS; for n ≤ 31 OpenBLAS is faster.  The MacBook Air throttles under
sustained load, so the measurements use a thermal guard; the
[benchmark campaign](benchmark_results/002-2026-10-24-macbook-m5-neon/README.md)
documents this together with hardware, software, methodology and the raw
data.

Where the remaining time goes was measured with an instrumented copy of
the GEMM driver: for n ≥ 1024, 97 to 98% of the time of a DGEMM call is
spent in the micro kernel at its own speed and 1.2 to 2.3% in packing,
which explains the small gap to BLIS.  For small n and large m and k the
packing of A took up to a quarter of the time; packing four panels of A
together if the columns of A are far apart made these shapes 5 to 9%
faster on the M5 and 2 to 13% on an Intel Core i5-12500
([profile campaign](benchmark_results/003-2026-10-25-macbook-m5-profile/README.md)).

### ZGEMM (complex double)

#### AArch64: SME2 and NEON on Apple M5, single-threaded ZGEMM

**Single-threaded ZGEMM on one performance core** of an Apple M5
(MacBook Air), m = n = k = 50, 100, …, 4000, complex double, alpha =
beta = 1: ulmBLAS (4fcbe80, Apple clang) with its own SME2 (16 x 16) and
NEON (4 x 3) ZGEMM kernels, against Apple Accelerate
(`VECLIB_MAXIMUM_THREADS=1`), OpenBLAS 0.3.34 and Eigen 5.0.1.  Medians
of three interleaved rounds under a thermal guard:

![Single-threaded ZGEMM n = 50 ... 4000 on an Apple M5: ulmBLAS SME2 and NEON, Apple Accelerate, Eigen and OpenBLAS](benchmark_results/029-2026-10-28-macbook-m5-zgemm/plots/zgemm_full.svg)

| n = m = k | ulmBLAS SME2 | ulmBLAS NEON | Accelerate | Eigen | OpenBLAS | SME2 / Accelerate |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 127.9 | 44.6 | 60.4 | 45.8 | 45.7 | 2.10 |
| 100 | 223.0 | 52.8 | 137.0 | 51.5 | 48.5 | 1.64 |
| 200 | 329.2 | 58.0 | 213.8 | 52.1 | 50.0 | 1.54 |
| 500 | 394.3 | 61.0 | 323.4 | 54.9 | 50.8 | 1.22 |
| 1000 | 417.6 | 60.5 | 359.0 | 54.1 | 50.7 | 1.16 |
| 2000 | 423.0 | 59.2 | 373.6 | 53.2 | 49.2 | 1.13 |
| 4000 | 412.9 | 57.7 | 372.1 | 51.8 | 48.2 | 1.11 |

(GFLOP/s, medians of 3 rounds; ratio: median of the per-round ratios.)

- **The SME2 kernel is faster than Accelerate at every size**: 1.123 for
  n ≥ 1000 (393 to 428 GFLOP/s), 1.64 for n = 50 … 250, at least 1.04 in
  every round.
- **Without SME2** ulmBLAS's NEON kernel is 1.20 times as fast as
  OpenBLAS and 1.11 times as fast as Eigen for n ≥ 1000.
- 1200 measurements, no wrong result; 58 runs were discarded by the
  thermal guard and repeated.  The MacBook was not otherwise idle (load
  average up to 3.5).

Details in the
[benchmark campaign](benchmark_results/029-2026-10-28-macbook-m5-zgemm/README.md).

#### x86-64: AVX-512 on Intel Xeon (Ice Lake), single-threaded ZGEMM

**Single-threaded ZGEMM on one core** of an Intel Xeon Platinum 8358
(Ice Lake-SP, AVX-512; bwUniCluster 3.0, one exclusive node), m = n = k =
50, 100, …, 4000, complex double, column major, alpha = beta = 1, 8 m n k
flops per call.  ulmBLAS (0e75bee, GCC 14.2) with five vendored ZGEMM
micro kernels and its reference kernel, against oneMKL 2025.0.1
(sequential), AOCL-BLAS 5.0, OpenBLAS 0.3.34 (SKYLAKEX) and BLIS 2.0
(skx and haswell).  Medians of three interleaved rounds:

![Single-threaded ZGEMM n = 50 ... 4000 on an Intel Xeon Platinum 8358: ulmBLAS with five micro kernels, oneMKL, AOCL-BLAS, OpenBLAS and BLIS](benchmark_results/028-2026-10-28-uc3-icelake-zgemm/plots/zgemm_full.svg)

The best ulmBLAS build uses **AOCL-BLAS's `zen4` 12 x 4 kernel**
(AVX-512), vendored into ulmBLAS
([ulmblas/level3/ukernel/blis](ulmblas/level3/ukernel/blis/README.md#zgemm-kernels)).
Native AOCL-BLAS selects the same kernel on this CPU; BLIS's `skx`
configuration has no complex kernel and computes ZGEMM with its DGEMM
kernel (induced method 1m).

| n = m = k | ulmBLAS + AOCL zen4 kernel | oneMKL | AOCL-BLAS | OpenBLAS | BLIS (skx, 1m) | ulmBLAS / oneMKL | ulmBLAS / AOCL-BLAS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 58.0 | 72.0 | 43.9 | 60.9 | 43.3 | 0.81 | 1.32 |
| 100 | 79.3 | 85.2 | 47.4 | 74.0 | 60.9 | 0.93 | 1.67 |
| 200 | 88.0 | 92.5 | 65.9 | 78.2 | 71.3 | 0.95 | 1.34 |
| 500 | 96.4 | 97.9 | 96.5 | 83.0 | 78.7 | 0.98 | 1.00 |
| 1000 | 96.7 | 99.0 | 96.4 | 83.4 | 78.9 | 0.98 | 1.00 |
| 2000 | 98.5 | 99.2 | 98.6 | 84.6 | 80.8 | 0.99 | 1.00 |
| 4000 | 98.4 | 101.6 | 99.0 | 85.0 | 80.5 | 0.97 | 0.99 |

(GFLOP/s, medians of 3 rounds; ratios: medians of the per-round ratios.)

- **Large matrices** (n ≥ 1000, 61 sizes): ulmBLAS with AOCL's kernel
  reaches 95.5 to 98.6 GFLOP/s: 0.970 of oneMKL, 0.992 of native
  AOCL-BLAS, 1.153 of OpenBLAS and 1.206 of BLIS.  OpenBLAS's SKYLAKEX
  kernel inside ulmBLAS reaches 0.987 of native OpenBLAS.
- **Small matrices**: native AOCL-BLAS is slow for n ≤ 200 on this Intel
  CPU; ulmBLAS with the same kernel is 1.26 to 1.67 times as fast there.
  oneMKL stays ahead (n = 50: ulmBLAS at 0.81).
- **Correctness**: 2480 measurements, no wrong result (400 elements of C
  per size checked against long double sums).

The [benchmark campaign](benchmark_results/028-2026-10-28-uc3-icelake-zgemm/README.md)
documents the builds, the kernels each library uses, the methodology and
all results.

#### x86-64: AVX2/FMA on Intel Core i5-12500, single-threaded ZGEMM

**Single-threaded ZGEMM on one core** of an Intel Core i5-12500 (Alder
Lake, AVX2/FMA), same sizes and method: the ZGEMM kernels of OpenBLAS
(HASWELL 4 x 2), AOCL-BLAS (haswell 3 x 4) and BLIS (haswell 3 x 4, 4 x 3,
sandybridge 4 x 4) inside ulmBLAS, against native OpenBLAS 0.3.34
(HASWELL) and BLIS 2.0 (haswell).

| n = m = k | ulmBLAS + OpenBLAS kernel | ulmBLAS + BLIS kernel | OpenBLAS | BLIS | ulmBLAS + OpenBLAS kernel / OpenBLAS | ulmBLAS + BLIS kernel / BLIS |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 50.8 | 45.6 | 56.2 | 44.3 | 0.90 | 1.03 |
| 100 | 58.3 | 55.1 | 59.3 | 55.0 | 0.98 | 1.00 |
| 200 | 58.9 | 58.0 | 59.7 | 57.9 | 0.99 | 1.00 |
| 500 | 61.2 | 59.8 | 61.3 | 59.7 | 1.00 | 1.00 |
| 1000 | 61.1 | 59.1 | 61.6 | 58.8 | 0.99 | 1.00 |
| 2000 | 61.9 | 60.0 | 62.1 | 59.6 | 1.00 | 1.01 |
| 4000 | 62.2 | 60.1 | 62.5 | 60.0 | 1.00 | 1.00 |

(GFLOP/s; "BLIS kernel": `bli_zgemm_haswell_asm_3x4`, the kernel native
BLIS uses.)  For n ≥ 1000 every kernel inside ulmBLAS reaches the library
it comes from: OpenBLAS's 0.995 of OpenBLAS, BLIS's 3 x 4 1.003 of BLIS;
1760 measurements, no wrong result
([campaign 027](benchmark_results/027-2026-10-28-heim-zgemm/README.md)).

### SGEMM (float)

#### x86-64: AVX-512 on Intel Xeon (Ice Lake), single-threaded SGEMM

**Single-threaded SGEMM on one core** of an Intel Xeon Platinum 8358
(Ice Lake-SP, AVX-512; bwUniCluster 3.0, one exclusive node), m = n = k =
50, 100, …, 4000, float, column major, alpha = beta = 1, 2 m n k flops
per call.  ulmBLAS (8a39314, GCC 14.2) with ten vendored SGEMM micro
kernels and its reference kernel, against oneMKL 2025.0.1 (sequential),
OpenBLAS 0.3.34 (SKYLAKEX), AOCL-BLAS 5.0 and BLIS 2.0 (skx and haswell).
Medians of three interleaved rounds:

![Single-threaded SGEMM n = 50 ... 4000 on an Intel Xeon Platinum 8358: ulmBLAS with ten micro kernels, oneMKL, OpenBLAS, AOCL-BLAS and BLIS](benchmark_results/030-2026-10-28-uc3-icelake-sgemm/plots/sgemm_full.svg)

| n = m = k | ulmBLAS + OpenBLAS SKYLAKEX kernel | ulmBLAS + BLIS skx kernel | oneMKL | OpenBLAS | BLIS (skx) | AOCL-BLAS | ulmBLAS (OpenBLAS kernel) / oneMKL | / OpenBLAS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 59.3 | 59.3 | 92.3 | 92.2 | 28.1 | 96.5 | 0.64 | 0.64 |
| 100 | 98.8 | 91.6 | 125.1 | 135.5 | 79.5 | 136.9 | 0.79 | 0.73 |
| 200 | 135.7 | 130.5 | 161.0 | 147.2 | 122.6 | 117.2 | 0.84 | 0.92 |
| 500 | 154.4 | 152.6 | 170.0 | 162.6 | 149.8 | 175.0 | 0.91 | 0.95 |
| 1000 | 168.5 | 161.1 | 190.9 | 172.4 | 160.5 | 159.6 | 0.88 | 0.98 |
| 2000 | 176.0 | 165.7 | 191.4 | 173.1 | 165.2 | 163.4 | 0.92 | 1.02 |
| 4000 | 180.7 | 166.0 | 193.8 | 178.3 | 163.5 | 164.4 | 0.93 | 1.01 |

(GFLOP/s, medians of 3 rounds; ratios: medians of the per-round ratios.)

- **Large matrices** (n ≥ 1000, 61 sizes)
  - OpenBLAS's SKYLAKEX kernel inside ulmBLAS: 168.5 to 180.7 GFLOP/s,
    1.007 of native OpenBLAS, 0.925 of oneMKL
  - BLIS's skx kernel inside ulmBLAS: 1.010 of native BLIS (same kernel
    and block sizes); AOCL-BLAS uses the same kernel on this CPU
  - the AVX2 kernels reach about half: BLIS haswell 6 x 16 inside ulmBLAS
    0.995 of native BLIS haswell
- **Small matrices**: oneMKL and OpenBLAS are ahead for n ≤ 200 (ulmBLAS
  0.64 of them at n = 50); native BLIS skx is slowest there (28 GFLOP/s
  at n = 50)
- **Correctness**: 3680 measurements, no wrong result

The [benchmark campaign](benchmark_results/030-2026-10-28-uc3-icelake-sgemm/README.md)
has all ten kernels, the block sizes and the method.

#### x86-64: AVX2/FMA on Intel Core i5-12500, single-threaded SGEMM

**Single-threaded SGEMM on one core** of an Intel Core i5-12500 (Alder
Lake, AVX2/FMA), same sizes and method as ZGEMM there: the SGEMM kernels of
OpenBLAS (HASWELL 8 x 4), AOCL-BLAS (haswell 6 x 16) and BLIS (haswell
6 x 16, 16 x 6, 4 x 24, 24 x 4, sandybridge 8 x 8, penryn 8 x 4) inside
ulmBLAS (47bd8f1), against native OpenBLAS 0.3.34 (HASWELL) and BLIS 2.0
(haswell).

| n = m = k | ulmBLAS + OpenBLAS kernel | ulmBLAS + BLIS kernel | OpenBLAS | BLIS | ulmBLAS + OpenBLAS kernel / OpenBLAS | ulmBLAS + BLIS kernel / BLIS |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 75.2 | 59.3 | 93.3 | 80.6 | 0.81 | 0.73 |
| 100 | 101.5 | 87.4 | 106.1 | 109.7 | 0.96 | 0.80 |
| 200 | 116.0 | 106.0 | 116.7 | 120.4 | 0.99 | 0.88 |
| 500 | 119.0 | 114.8 | 119.4 | 113.9 | 1.00 | 1.01 |
| 1000 | 122.9 | 119.3 | 122.6 | 118.9 | 1.00 | 1.00 |
| 2000 | 122.7 | 120.3 | 123.3 | 119.8 | 1.00 | 1.00 |
| 4000 | 123.5 | 120.6 | 124.2 | 121.0 | 0.99 | 1.00 |

(GFLOP/s; "BLIS kernel": `bli_sgemm_haswell_asm_6x16`, the kernel native
BLIS uses.)  For n ≥ 1000 every kernel inside ulmBLAS reaches the library
it comes from: OpenBLAS's 0.992 of OpenBLAS, BLIS's 6 x 16 1.002 of BLIS.
For n ≤ 200 both native libraries are faster.  2480 measurements, no
wrong result
([campaign 033](benchmark_results/033-2026-10-28-heim-sgemm/README.md)).

#### AArch64: SME2 and NEON on Apple M5, single-threaded SGEMM

**Single-threaded SGEMM on one performance core** of an Apple M5
(MacBook Air), m = n = k = 50, 100, …, 4000, float, alpha = beta = 1:
ulmBLAS (9cf736f, Apple clang) with its own SME2 (32 x 32) and NEON
(8 x 12) SGEMM kernels, against Apple Accelerate
(`VECLIB_MAXIMUM_THREADS=1`), OpenBLAS 0.3.34 and Eigen 5.0.1.  Medians
of three interleaved rounds under a thermal guard:

![Single-threaded SGEMM n = 50 ... 4000 on an Apple M5: ulmBLAS SME2 and NEON, Apple Accelerate, Eigen and OpenBLAS](benchmark_results/032-2026-10-29-macbook-m5-sgemm/plots/sgemm_full.svg)

| n = m = k | ulmBLAS SME2 | ulmBLAS NEON | Accelerate | Eigen | OpenBLAS | SME2 / Accelerate |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 340.0 | 64.5 | 347.3 | 93.1 | 89.5 | 0.97 |
| 100 | 567.6 | 90.1 | 715.7 | 108.7 | 101.9 | 0.82 |
| 200 | 973.8 | 110.0 | 1233.6 | 115.1 | 105.9 | 0.78 |
| 500 | 1446.9 | 118.5 | 1560.1 | 118.1 | 109.0 | 0.96 |
| 1000 | 1535.8 | 121.1 | 1705.7 | 119.6 | 108.9 | 0.89 |
| 2000 | 1623.4 | 123.5 | 1600.4 | 120.0 | 110.6 | 1.02 |
| 4000 | 1615.9 | 121.5 | 1557.1 | 116.9 | 108.3 | 1.04 |

(GFLOP/s, medians of 3 rounds; ratio: median of the per-round ratios.)

- **Large matrices: the SME2 kernel is about as fast as Accelerate**,
  1.023 for n ≥ 1000 (1416 to 1629 GFLOP/s).  Accelerate is faster up to
  n = 1400 (0.92 of it there); at n = 1450 Accelerate drops by 16%, and
  from there on the SME2 kernel is ahead (1.026, at 46 of 52 sizes).
- **Small matrices: Accelerate is faster**, the SME2 kernel reaches 0.86
  of it for n = 50 … 250.  Unlike ZGEMM (1.64 there), SGEMM has no lead
  for small sizes.
- **Without SME2** ulmBLAS's NEON kernel is 1.12 times as fast as
  OpenBLAS and 1.04 times as fast as Eigen for n ≥ 1000.
- 1200 measurements, no wrong result; 1 run was discarded by the thermal
  guard and repeated.

Details in the
[benchmark campaign](benchmark_results/032-2026-10-29-macbook-m5-sgemm/README.md).

### CGEMM (complex float)

#### x86-64: AVX-512 on Intel Xeon (Ice Lake), single-threaded CGEMM

**Single-threaded CGEMM on one core** of an Intel Xeon Platinum 8358
(Ice Lake-SP, AVX-512; bwUniCluster 3.0, one exclusive node), m = n = k =
50, 100, …, 4000, complex float, column major, alpha = beta = 1, 8 m n k
flops per call.  ulmBLAS (8a39314, GCC 14.2) with six vendored CGEMM
micro kernels and its reference kernel, against oneMKL 2025.0.1
(sequential), OpenBLAS 0.3.34 (SKYLAKEX), AOCL-BLAS 5.0 and BLIS 2.0 (skx
and haswell).  Medians of three interleaved rounds:

![Single-threaded CGEMM n = 50 ... 4000 on an Intel Xeon Platinum 8358: ulmBLAS with six micro kernels, oneMKL, OpenBLAS, AOCL-BLAS and BLIS](benchmark_results/031-2026-10-29-uc3-icelake-cgemm/plots/cgemm_full.svg)

| n = m = k | ulmBLAS + OpenBLAS SKYLAKEX kernel | ulmBLAS + AOCL haswell kernel | oneMKL | OpenBLAS | BLIS (skx, 1m) | AOCL-BLAS | ulmBLAS (OpenBLAS kernel) / oneMKL | / OpenBLAS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 91.9 | 62.8 | 127.8 | 107.3 | 65.2 | 54.5 | 0.72 | 0.86 |
| 100 | 130.2 | 81.5 | 164.2 | 139.3 | 113.7 | 95.8 | 0.79 | 0.93 |
| 200 | 158.6 | 93.2 | 181.7 | 161.0 | 143.0 | 96.3 | 0.87 | 0.98 |
| 500 | 168.2 | 96.2 | 190.2 | 171.8 | 157.8 | 96.8 | 0.88 | 0.98 |
| 1000 | 175.7 | 98.3 | 199.8 | 176.7 | 163.6 | 98.7 | 0.88 | 0.99 |
| 2000 | 178.4 | 98.7 | 197.3 | 178.5 | 164.0 | 99.0 | 0.90 | 1.00 |
| 4000 | 181.6 | 99.6 | 204.5 | 181.8 | 163.2 | 99.8 | 0.89 | 1.00 |

(GFLOP/s, medians of 3 rounds; ratios: medians of the per-round ratios.)

- **Large matrices** (n ≥ 1000, 61 sizes)
  - OpenBLAS's SKYLAKEX kernel inside ulmBLAS: 172.9 to 181.6 GFLOP/s,
    0.997 of native OpenBLAS, 0.888 of oneMKL
  - BLIS and AOCL-BLAS have no AVX-512 complex float kernel: BLIS skx
    computes CGEMM with its SGEMM kernel (1m, 0.92 of OpenBLAS), AOCL-BLAS
    uses the AVX2 kernel haswell 3x8 on this CPU, so OpenBLAS's kernel
    inside ulmBLAS is 1.80 times AOCL-BLAS
  - the BLIS and AOCL haswell kernels inside ulmBLAS reach 0.994 and 0.996
    of their native library
- **Correctness**: 2720 measurements, no wrong result

The [benchmark campaign](benchmark_results/031-2026-10-29-uc3-icelake-cgemm/README.md)
has all six kernels, the block sizes and the method.

#### AArch64: SME2 and NEON on Apple M5, single-threaded CGEMM

**Single-threaded CGEMM on one performance core** of an Apple M5
(MacBook Air), m = n = k = 50, 100, …, 4000, complex float,
alpha = beta = 1: ulmBLAS (9cf736f, Apple clang) with its own SME2
(32 x 16) and NEON (4 x 6) CGEMM kernels, against Apple Accelerate
(`VECLIB_MAXIMUM_THREADS=1`), OpenBLAS 0.3.34 and Eigen 5.0.1.  Medians
of three interleaved rounds under a thermal guard:

![Single-threaded CGEMM n = 50 ... 4000 on an Apple M5: ulmBLAS SME2 and NEON, Apple Accelerate, Eigen and OpenBLAS](benchmark_results/035-2026-10-29-macbook-m5-cgemm/plots/cgemm_full.svg)

| n = m = k | ulmBLAS SME2 | ulmBLAS NEON | Accelerate | Eigen | OpenBLAS | SME2 / Accelerate |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 305.7 | 85.6 | 127.0 | 91.9 | 85.0 | 2.41 |
| 100 | 538.0 | 107.5 | 300.0 | 101.7 | 93.9 | 1.81 |
| 200 | 926.8 | 115.2 | 614.2 | 107.4 | 97.2 | 1.49 |
| 500 | 1385.5 | 121.1 | 1086.3 | 107.9 | 99.2 | 1.28 |
| 1000 | 1514.6 | 123.7 | 1360.0 | 108.5 | 98.9 | 1.12 |
| 2000 | 1572.4 | 122.3 | 1473.6 | 106.8 | 98.4 | 1.06 |
| 4000 | 1562.0 | 119.3 | 1475.7 | 104.8 | 96.1 | 1.06 |

(GFLOP/s, 8mnk flops, medians of 3 rounds; ratio: median of the
per-round ratios.)

- **The SME2 kernel is faster than Accelerate for all sizes but one**:
  1.065 for n ≥ 1000 (1336 to 1591 GFLOP/s); only at n = 1050 is
  Accelerate ahead (0.996).
- **Small matrices: the SME2 kernel is much faster**, 1.67 times
  Accelerate for n = 50 … 250, 1.27 for n = 300 … 500.  As for ZGEMM, and
  unlike SGEMM.
- **Without SME2** ulmBLAS's NEON kernel is 1.24 times as fast as
  OpenBLAS and 1.14 times as fast as Eigen for n ≥ 1000.
- 1200 measurements, no wrong result; 5 runs were discarded by the
  thermal guard and repeated.

Details in the
[benchmark campaign](benchmark_results/035-2026-10-29-macbook-m5-cgemm/README.md).

#### x86-64: AVX2/FMA on Intel Core i5-12500, single-threaded CGEMM

**Single-threaded CGEMM on one core** of an Intel Core i5-12500 (Alder
Lake, AVX2/FMA), same sizes and method: the CGEMM kernels of OpenBLAS
(HASWELL 8 x 2), AOCL-BLAS (haswell 3 x 8) and BLIS (haswell 3 x 8, 8 x 3,
sandybridge 8 x 4) inside ulmBLAS (47bd8f1), against native OpenBLAS
0.3.34 (HASWELL) and BLIS 2.0 (haswell).

| n = m = k | ulmBLAS + OpenBLAS kernel | ulmBLAS + BLIS kernel | OpenBLAS | BLIS | ulmBLAS + OpenBLAS kernel / OpenBLAS | ulmBLAS + BLIS kernel / BLIS |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 82.7 | 78.1 | 109.2 | 72.2 | 0.76 | 1.08 |
| 100 | 102.0 | 100.6 | 118.0 | 100.3 | 0.86 | 1.00 |
| 200 | 118.9 | 116.0 | 121.9 | 115.9 | 0.98 | 1.00 |
| 500 | 120.6 | 118.8 | 123.4 | 118.4 | 0.98 | 1.00 |
| 1000 | 122.4 | 119.9 | 123.7 | 120.0 | 0.99 | 1.00 |
| 2000 | 123.5 | 120.9 | 123.8 | 119.9 | 1.00 | 1.01 |
| 4000 | 124.5 | 122.0 | 124.7 | 121.2 | 1.00 | 1.01 |

(GFLOP/s; "BLIS kernel": `bli_cgemm_haswell_asm_3x8`, the kernel native
BLIS uses.)  For n ≥ 1000 OpenBLAS's kernel inside ulmBLAS reaches 0.992
of OpenBLAS, BLIS's 3 x 8 1.001 of BLIS; for small matrices OpenBLAS is
ahead (0.76 at n = 50).  1760 measurements, no wrong result
([campaign 034](benchmark_results/034-2026-10-29-heim-cgemm/README.md)).

## Symmetric matrices: dsymm, dsyrk

C ← alpha A B + beta C with symmetric A (dsymm) and C ← alpha A Aᵀ +
beta C (dsyrk); n = 2048, GFLOP/s.

| Routine | heim: ulmBLAS AVX/FMA | BLIS 2.0 | OpenBLAS 0.3.34 | M5: ulmBLAS NEON | BLIS 2.0 | OpenBLAS 0.3.34 |
|---|---:|---:|---:|---:|---:|---:|
| dsymm | 58.87 | 58.16 | 59.71 | 64.12 | 66.44 | 60.60 |
| dsyrk | 50.48 | 53.93 | 57.59 | 62.72 | 63.98 | 59.07 |

With the SME2 backend on the M5 (n = 512, 1024, 2048): dsymm 413 … 432,
dsyrk 315 … 381 GFLOP/s, about BLIS pull request #921; Apple Accelerate
is faster for n = 512.  Campaigns:
[001](benchmark_results/001-2026-10-24-heim-avx-fma/README.md) (heim),
[002](benchmark_results/002-2026-10-24-macbook-m5-neon/README.md) (M5
NEON), [004](benchmark_results/004-2026-10-25-macbook-m5-sme/README.md)
(M5 SME2), [008](benchmark_results/008-2026-10-26-heim-avx-fma-memory/README.md)
(heim, later revision).

![Level 3 routines on heim: SSE, AVX/FMA, BLIS and OpenBLAS](benchmark_results/001-2026-10-24-heim-avx-fma/plots/level3.svg)

## Triangular matrices: dtrmm, dtrsm

B ← alpha A B with triangular A (dtrmm) and the solution of A X = alpha
B (dtrsm); n = 2048, GFLOP/s.

| Routine | heim: ulmBLAS AVX/FMA | BLIS 2.0 | OpenBLAS 0.3.34 | M5: ulmBLAS NEON | BLIS 2.0 | OpenBLAS 0.3.34 |
|---|---:|---:|---:|---:|---:|---:|
| dtrmm | 56.53 | 55.74 | 57.10 | 64.73 | 64.60 | 59.60 |
| dtrsm | 53.05 | 55.11 | 55.76 | 61.07 | 61.59 | 57.65 |

With the SME2 backend: dtrmm 344 … 401 GFLOP/s, but dtrsm only 67 … 176
(Accelerate 241 … 284): the triangular solve inside the diagonal blocks
uses the scalar reference kernel.  Campaigns as above.

![Level 3 routines on an Apple M5 with SME2](benchmark_results/004-2026-10-25-macbook-m5-sme/plots/level3.svg)
