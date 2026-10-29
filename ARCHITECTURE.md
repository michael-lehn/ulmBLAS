# How ulmBLAS is organized

This file is for you if you want to understand how ulmBLAS works inside,
add a micro kernel, or measure it yourself.  It starts with the algorithm
that all level 3 routines share, then goes through the source tree, the
micro kernels and how to add one, and ends with the interfaces, the tests
and the benchmarks.

- [The GEMM algorithm](#the-gemm-algorithm)
- [The source tree](#the-source-tree)
- [Micro kernels and platforms](#micro-kernels-and-platforms)
- [Adding a micro kernel](#adding-a-micro-kernel)
- [Interfaces](#interfaces)
- [Testing](#testing)
- [Benchmarks](#benchmarks): [running your own](#how-to-run-your-own-benchmarks), [the benchmark archive](#benchmark-archive)

This document is still rough, and I will polish it in the future.  In
the meantime have a look at my lecture
[High Performance Computing I (winter 2025/26)](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/), where we build
exactly this step by step, with more detailed explanations and
exercises:

| Step | Lecture page |
|---|---|
| Loop order and the cache | [Simple cache optimizations](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session04/page01.html) |
| The naive matrix product and its loop variants | [GEMM: general matrix-matrix product](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session05/page01.html) |
| Packing blocks of A and B into buffers | [Packing blocks of A](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session09/page01.html), [Packing blocks of B](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session09/page03.html) |
| Micro kernel, macro kernel and the blocked algorithm | [GEMM micro kernel](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session10/page01.html), [GEMM macro kernel](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session10/page03.html), [GEMM frame operation](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session10/page05.html), [Benchmark suite](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session10/page07.html) |
| Faster micro kernels and the comparison with MKL | [Status quo](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session19/page01.html), [Using the GCC vector extensions](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session19/page02.html), [Using the micro kernel from BLIS](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session19/page04.html), [Comparison with Intel MKL](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/session19/page06.html) |

The AVX/FMA micro kernel of ulmBLAS is the BLIS kernel as adapted in
this lecture.

## The GEMM algorithm

All level 3 routines share the structure of the BLIS GEMM algorithm.  In
its simplest form, and that is how I implemented it in 2014, it works
like this.  For C ← β C + α A B the operands are partitioned into blocks
that fit into the caches, and the blocks are copied ("packed") into
contiguous panels in the order in which the micro kernel reads them:

```
for jc = 0, NC, 2 NC, ...                    columns of C and B
  for pc = 0, KC, 2 KC, ...                  rank-KC updates
    pack B(pc:pc+KC, jc:jc+NC) -> B~         KC x NC, panels of NR columns (L3)
    for ic = 0, MC, 2 MC, ...
      pack A(ic:ic+MC, pc:pc+KC) -> A~       MC x KC, panels of MR rows    (L2)
      macro kernel:
        for each MR x NR tile of C(ic:ic+MC, jc:jc+NC):
          micro kernel:  C_tile <- beta C_tile + alpha * (MR x KC panel of A~)
                                                         * (KC x NR panel of B~)
```

The micro kernel keeps the MR x NR tile in registers and performs
2 · MR · NR · KC floating-point operations on data that streams from the
L1 cache (B~ panel) and the L2 cache (A~ panel).  How fast it is decides
how fast all level 3 routines are.

This is still the algorithm today.  Two later improvements change only
how the blocks are packed; the packed layout, and therefore the micro
kernels, stay the same.

**Packing several panels of A together.**  Packing A panel by panel reads
MR elements from each column of A per panel.  When the columns of A are
far apart in memory, this is slow.  On the Apple M5, packing A took up to
26% of the time of DGEMM with small n (2048 x 64 x 2048).  I found this
while profiling DGEMM during the Apple M5 experiments
([campaign 003](benchmark_results/003-2026-10-25-macbook-m5-profile/README.md)).
Now `gepack_A` packs `PACK_A_PANELS` panels (4) at once, column by
column, if the columns of A are at least `PACK_A_GROUP_DISTANCE` bytes
(8 KiB) apart.  Small n with m, k ≥ 1024 got 5 to 9% faster on the M5 and
2 to 13% on an Intel Core i5-12500; for matrices already in the cache
grouping would be slower, hence the threshold.  Both parameters are set
per data type in `ulmblas/config/blocksize.h`.

**Packing B interleaved with the first block of A.**  The algorithm above
packs a whole KC x NC block of B before it computes anything.  OpenBLAS's
driver does it differently: for the first block of A it packs B panel by
panel and uses each panel right away, while it is still in the cache.  I
adopted this when I tuned OpenBLAS's micro kernel inside ulmBLAS on an
Intel Xeon (Ice Lake), where it made DGEMM 0.8% faster for n ≥ 1000
([campaign 021](benchmark_results/021-2026-10-27-uc3-icelake-openblas-kernel-interleave/README.md)).
It is used only with the OpenBLAS-derived DGEMM kernels
(`-DULMBLAS_DGEMM_NO_INTERLEAVE` restores the original order):

```
for jc ...
  for pc ...
    for ic = 0:                               first block of A
      pack A(ic:ic+MC, pc:pc+KC) -> A~
      for jr = 0, NR, 2 NR, ...               panel by panel
        pack B(pc:pc+KC, jc+jr:jc+jr+NR) -> B~ panel
        macro kernel on this panel
    for ic = MC, 2 MC, ...                    the other blocks of A
      pack A -> A~, macro kernel on all of B~ (as above)
```

## The source tree

The code is organized in layers, from the interfaces at the top to the
micro kernels and the configuration at the bottom:

| Layer | Directory | Responsibility |
|---|---|---|
| BLAS interfaces | `interfaces/blas/C`, `interfaces/blas/F77`, `interfaces/lapack/C` | argument checks, `xerbla`, row major as transposed column major |
| Level 3 algorithms | `ulmblas/level3/*.tcc` | cache blocking (MC, KC, NC), structure of symmetric, Hermitian and triangular matrices |
| Packing | `ulmblas/level3/pack` | copying blocks into MR and NR panels, zero padding, symmetric and triangular blocks |
| Macro kernels | `ulmblas/level3/mkernel` | loops over the MR x NR tiles; edge tiles through a buffer; tiles on the diagonal of triangular results |
| Micro kernels | `ulmblas/level3/ukernel/{ref,sse,avx,neon,sme,blis,openblas}` | the MR x NR update, the architecture-specific part (the SME2 backend also packs single panels) |
| Configuration | `ulmblas/config` | kernel selection, block sizes, compile-time checks |

The macro kernels work with *any* register block size: the edge tiles of a
block (mr < MR or nr < NR) are computed by the micro kernel into a local
buffer, and the diagonal tiles of syrk, syr2k, herk and her2k are found by
comparing row and column ranges.  MR and NR need not divide each other (the
AVX/FMA and NEON DGEMM kernels use MR = 6, NR = 8), and a test configuration
runs the reference kernel with these block sizes on every platform.  Static checks
enforce the remaining constraints (MC a multiple of MR and NR, NC a multiple
of NR, so that the packed panels fit the buffers).

Two exceptions concern small problems.  On AArch64 (NEON or SME2), `gemm`
for double computes problems with m·n·k ≤ 8192 (A and C with unit row
stride, as column major and the Fortran interface pass them) directly
with NEON register tiles on A, B and C, without packing (`ulmblas/level3/gemmdirect.tcc`;
the threshold is `ULMBLAS_GEMM_DIRECT_MAX_MNK`, and 0 disables the path).
The packing buffers are kept between calls in a small cache per thread
(`ulmblas/auxiliary/memorypool.tcc`): no lock, no shared state, and a
block released by another thread is simply freed.

## Micro kernels and platforms

| Kernel | Data types | MR x NR | Instruction set | Selected |
|---|---|---|---|---|
| reference (`ukernel/ref`) | all four | 4 x 4 (double; 8 x 4, 4 x 2, 2 x 2 for float, complex float, complex double) | portable C++ | on x86 without SSE3, on AArch64 with `-DULMBLAS_NO_NEON`, on all other targets |
| SSE3 (`ukernel/sse`) | double: GEMM and TRSM kernels; level 1 kernels | 4 x 4 | x86-64 SSE3, inline assembly | by default on x86-64 |
| AVX/FMA (`ukernel/avx`) | double: GEMM kernel | 6 x 8 | x86-64 AVX and FMA, inline assembly derived from BLIS | with `ARCHFLAGS='-mavx -mfma'` (or e.g. `-march=native`) |
| NEON (`ukernel/neon`) | double: GEMM kernel (derived from BLIS); float, complex float, complex double: GEMM kernels | 6 x 8; 8 x 12, 4 x 6, 4 x 3 | AArch64 AdvSIMD (NEON) FMA, inline assembly and intrinsics | by default on AArch64 (64-bit pointers) |
| SME2, experimental (`ukernel/sme`) | double: GEMM kernel (derived from BLIS pull request #921) and panel packing; float, complex float, complex double: GEMM kernels and packing | 32 x 16; 32 x 32, 32 x 16, 16 x 16 | AArch64 SME2 (FMOPA into ZA, streaming mode), ACLE intrinsics | with clang and flags enabling SME2, e.g. `ARCHFLAGS=-mcpu=apple-m4` |

Kernels are selected at compile time from the compiler's target macros
(`ulmblas/config/simd.h`); there is no runtime dispatch, so a library built
with AVX/FMA requires a CPU with AVX and FMA.  AdvSIMD is part of every AArch64
CPU, so the NEON kernel needs no flags; `CPPFLAGS=-DULMBLAS_NO_NEON` selects
the reference kernels.  The SME2 backend replaces the NEON kernels for all four
types if the compiler is clang and the flags enable SME2;
`CPPFLAGS=-DULMBLAS_NO_SME` keeps the NEON kernels.  Types without an
optimized kernel use the reference kernel.  The vendored kernels of BLIS,
AOCL-BLAS and OpenBLAS (x86-64) are selected explicitly, per data type,
with `-DULMBLAS_DGEMM_UKERNEL_…`, `-DULMBLAS_ZGEMM_UKERNEL_…`,
`-DULMBLAS_SGEMM_UKERNEL_…` and `-DULMBLAS_CGEMM_UKERNEL_…` (see the
[support matrix](README.md#current-micro-kernel-support) and
[ulmblas/level3/ukernel/blis](ulmblas/level3/ukernel/blis/README.md)).

| Platform | Built | Test suite run | Optimized kernels executed and benchmarked |
|---|---|---|---|
| Linux x86-64, Debian 12, Intel Core i5-12500; GCC 12, clang 16 | yes | full `make check` (the NEON and SME2 configurations are skipped) | SSE3 and AVX/FMA executed, both benchmarked |
| macOS arm64, Apple M5; Apple clang 21, GCC 16 | yes | full `make check` (the AVX/FMA configurations are skipped; with GCC also the SME2 configuration) | NEON and SME2 (Apple clang) executed and benchmarked |
| macOS x86-64 under Rosetta 2; Apple clang 21 | yes | numeric tests (SSE3 kernels) | SSE3 executed, not benchmarked; AVX/FMA only compiled (Rosetta 2 provides no AVX) |
| 32-bit x86 | compiled (all interface sources; reference kernels), not linked | no | — |

Other platforms with a C++11 compiler and GNU Make should work with the
reference kernels, but I have not tried them.  Optimized kernels exist for
x86-64 and AArch64.  The NEON kernels have only run on Apple silicon (Linux
on AArch64 is untested), the SME2 backend only on an Apple M5.  The
x86-64 vendored kernels were also tested on an Intel Xeon (Ice Lake) and,
for DGEMM, on an AMD EPYC (Zen 4) of bwUniCluster 3.0.  More in
[BUILDING.md](BUILDING.md#tested-configurations).

Every kernel, with origin, files, selection macro, requirements, tests and
benchmarks, is listed in the
[micro kernel README](ulmblas/level3/ukernel/README.md); the vendored
kernels of other libraries and how they are integrated are described in
[ulmblas/level3/ukernel/blis](ulmblas/level3/ukernel/blis/README.md) and
[ulmblas/level3/ukernel/openblas](ulmblas/level3/ukernel/openblas/README.md).

## Adding a micro kernel

A micro kernel for a new architecture or data type plugs into the existing
algorithms, and all routines that use the GEMM micro kernel (the complete
level 3 BLAS) benefit without further changes.  Kernels from BLIS are a
natural source, BLIS has kernels for many x86-64, ARM, POWER and RISC-V
processors, and the AVX/FMA and NEON kernels show how it is done.  You
can't just copy a BLIS kernel, though.  These are the things to take care
of:

- Interface.  ulmBLAS calls
  `ugemm(kc, alpha, A, B, beta, C, incRowC, incColC, nextA, nextB)` with
  references to alpha and beta and strides of C; BLIS kernels have a
  different signature and context arguments.
- Packing layout.  A is packed as MR x kc panels (column by column), B as
  kc x NR panels (row by row), as in BLIS; kernels expecting other layouts
  need adapted packing.  A kernel that stores its tile with unit column
  stride can compute the transposed product instead: the NEON kernel calls
  the BLIS 8 x 6 kernel with the panels of B and A swapped (C^T = B^T A^T).
- Block sizes.  MR, NR and the cache block sizes MC, KC, NC are set per
  kernel and data type in `ulmblas/config/blocksize.h`; MC and KC must be
  chosen (measured) for the cache sizes of the target.
- Edge tiles and strides.  ulmBLAS calls the kernel for full tiles only;
  edge tiles go through a buffer.  If the assembly code supports only unit
  row or column strides of C (as the BLIS kernel), the kernel needs a fallback
  for general strides.
- Numerical semantics.  With β = 0, C must not be read (it may contain
  NaN); β = NaN must not be taken as zero.
- Alignment and memory.  The packing buffers are aligned for the kernel
  (32 bytes for AVX) and padded for reading ahead; inline assembly must declare
  all modified registers and memory.
- Instruction set, compiler and object format.  The kernel is selected by
  the compiler's target macros; inline assembly must assemble with GCC and
  clang and work on ELF and Mach-O (e.g. numeric local labels).
- Tests.  A test configuration with `CONFIG_REQUIRES` runs the complete
  test suite with the new kernel where the CPU supports it and reports SKIP
  elsewhere; `test/numeric/ukernel.cc` tests the micro kernel directly.
- License and attribution.  The upstream copyright and license notices
  must be preserved (see [License](README.md#license-provenance-and-acknowledgments)).

The AVX/FMA kernel (`ulmblas/level3/ukernel/avx/ugemm.tcc`) is a good
example: it lists where the code comes from and every change I made to
the BLIS code.

## Interfaces

| Library | Header | Contents |
|---|---|---|
| `libcblas.a` | `cblas.h` | CBLAS (`cblas_dgemm`, ...): levels 1, 2 and 3 for float, double, complex float and complex double |
| `libclapack.a` | `clapack.h` | `ulmblas_dgetrf` (LU factorization with partial pivoting) and `ulmblas_dlaswp` |
| `libf77blas.a` | `f77blas.h` | Fortran 77 BLAS interface (`make f77blas`) |
| `libatlblas.a`, `libatllapack.a` | `atlblas.h`, `atllapack.h` | ATLAS names, used by the benchmarks |

Both BLAS interfaces implement the 148 routines of the reference BLAS
(BLAS/SRC of LAPACK 3.3.1, except the auxiliary routines `lsame` and
`xerbla_array`): the 142 routines of the CBLAS standard, the complex
Givens rotations `crotg`, `zrotg`, `csrot`, `zdrot` and the functions
`scabs1`, `dcabs1` (in the C interface with the names of the CBLAS of
LAPACK 3.12, e.g. `cblas_crotg`).  Level 1 routines without optimized
kernels (e.g. `rotm`, `rotmg`, `sdsdot`, `dsdot`) are portable reference
implementations.  Behavior follows the
reference BLAS of LAPACK 3.12.1, with the correction of the rescaling in
`?rotmg` of its development version; `i?amax` with NaN follows 3.12.1
(open question, see [`test/known_issues.md`](test/known_issues.md)).

Not provided: `lsame` and `xerbla_array` (auxiliary routines that LAPACK
contains itself), `?gemmtr` (LAPACK 3.12), the routines of the
Reference-LAPACK development version (`?axpby`, skew-symmetric routines).
The index type of all interfaces is `ULMBLAS_INDEX_TYPE` (default `int`,
64-bit types on request, see [INSTALL.md](INSTALL.md#index-type)); the
interfaces call the templates with `std::ptrdiff_t`.  The error handlers `cblas_xerbla` and
`xerbla_` print a message and return; the function then returns without
computing (reference BLAS stops the program).  The C++ templates in
`ulmblas/` are the implementation, not a stable public interface.

## Testing

`make check` runs, for several kernel and blocking configurations, a
regression test suite for every routine and data type, tests of the micro
kernels, the examples and the Netlib BLAS test suites.  I keep the
details of what is tested, the options and the known issues in one place,
the [Test section of INSTALL.md](INSTALL.md#test).  For a new kernel the
part that matters is the test configuration, see
[Adding a micro kernel](#adding-a-micro-kernel).

## Benchmarks

There are two ways to look at performance: you can run the benchmark of
ulmBLAS against the libraries on your own machine, or look at my
documented campaigns in the benchmark archive (summarized in
[PERFORMANCE.md](PERFORMANCE.md)).

### How to Run Your Own Benchmarks

`make bench-gemm` builds ulmBLAS (with your `ARCHFLAGS`) and measures
single-threaded GEMM, C ← A·B + C for square matrices m = n = k = 50,
100, …, 4000 (80 sizes), for ulmBLAS and for every other BLAS library it
finds on the machine.  The other libraries are optional, without them it
benchmarks ulmBLAS alone.  The 80 sizes take a few minutes per library
for fast libraries and real types, and more than an hour for slow ones
(e.g. a reference kernel for complex types).  More in
[bench/gemm/README.md](bench/gemm/README.md).

Only ulmBLAS:

```sh
make bench-gemm LIBS=ulmblas                    # DGEMM, 80 sizes
make bench-gemm LIBS=ulmblas SIZES="200 400" SAMPLES=1   # a quick check
```

The run prints where it writes its results (by default
`bench/gemm/results/<date>-<time>-dgemm/`).  You find `summary.md` with
GFLOP/s at a few sizes, `raw/ulmblas-1.txt` with median, minimum and
maximum GFLOP/s and the result check for every size, `detection.txt` with
the machine, the compilers and the micro kernels and block sizes ulmBLAS
was built with, and `gflops.svg` if Gnuplot is installed.

ulmBLAS and OpenBLAS:

```sh
make bench-gemm LIBS="ulmblas openblas"
```

OpenBLAS is found through pkg-config, Homebrew or the default linker path,
or in `OPENBLAS_DIR`.  A library only counts as found if the benchmark
program links, runs and reports it.  `detection.txt` and the output of
the run show what was used, e.g.

```
openblas: selected, found via Homebrew (/opt/homebrew/opt/openblas)
     cblas_dgemm from /opt/homebrew/Cellar/openblas/0.3.34/lib/libopenblasp-r0.3.34.dylib
     library OpenBLAS: OpenBLAS 0.3.34 DYNAMIC_ARCH NO_AFFINITY USE_OPENMP vortexm4 ...
```

If a library named in `LIBS` is not found, the run stops with an error.
Nothing is substituted for it.

All libraries that can be found:

```sh
make bench-gemm                                 # LIBS=auto: ulmBLAS and every library found
MKLROOT=/opt/intel/oneapi/mkl/latest make bench-gemm PREC=z   # e.g. with oneMKL, ZGEMM
```

| Library | `LIBS` name | Found |
|---|---|---|
| ulmBLAS (this tree) | `ulmblas` | always |
| OpenBLAS | `openblas` | automatically (pkg-config, Homebrew, linker path); or `OPENBLAS_DIR` |
| BLIS | `blis` | automatically (pkg-config, linker path); or `BLIS_DIR` |
| Apple Accelerate | `accelerate` | automatically on macOS |
| Eigen (native C++ product, no CBLAS) | `eigen` | automatically (pkg-config `eigen3`); or `EIGEN_DIR` |
| Intel oneMKL (sequential) | `mkl` | with `MKLROOT` (set by oneAPI's `setvars.sh`) |
| AOCL-BLAS | `aocl` | with `AOCL_DIR` or `AOCL_ROOT` (its library is also named `libblis`, so it is never taken for BLIS) |
| ATLAS, Netlib reference BLAS | — | not supported by `bench-gemm` (only by the older `make bench`, see [bench/](bench)) |

Libraries that are not found are skipped and listed with the reason.

Options: `PREC=s|d|c|z` selects SGEMM, DGEMM, CGEMM or ZGEMM;
`LIBS="..."` the libraries; `ARCHFLAGS='...'` the ulmBLAS micro kernels
(e.g. `ARCHFLAGS='-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_8x6'`,
see the [support matrix](README.md#current-micro-kernel-support)); `SAMPLES=n`
the timed samples per size (default 3, median reported); `ROUNDS=n`
repetitions over all libraries in rotated order (default 1); `SIZES="..."`
the sizes; `CPU=n` pins the programs to one CPU on Linux; `OUT=dir` the
result directory.  Every library runs with one thread (thread counts set
to 1, oneMKL linked sequential).  GFLOP/s count 2mnk floating-point
operations for SGEMM and DGEMM and 8mnk for CGEMM and ZGEMM.

The results depend on the CPU, the compiler, how each library was built,
the kernel it selects at run time and the threading configuration, so
measure on an otherwise idle machine.  My published results come from
documented campaigns with more control (pinning, load gate, interleaved
rounds), see the [benchmark archive](benchmark_results/README.md).

### Benchmark archive

In [benchmark_results/](benchmark_results/README.md) I collect benchmark
campaigns over time, on different processors and with different
compilers, micro kernels and blocking strategies.  Each campaign has the
raw measurements, CSV files, Gnuplot scripts and the SVG figures, the
benchmark programs, and a description of hardware, software, source
revisions and methodology.

A completed campaign directory is added with

```sh
make add_benchmark_result BENCHMARK_DIR=path/to/campaign   # needs Gnuplot
```

which checks the campaign, regenerates its figures and updates the index.
It never overwrites archived measurements and does not run benchmarks.  The
older ATLAS-based benchmark programs in `bench/` (`make bench`) are
described in [BUILDING.md](BUILDING.md#benchmarks).
