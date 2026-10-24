# ulmBLAS

**A BLAS implementation in C++ that makes the anatomy of high-performance
matrix multiplication explicit — and extensible.**

ulmBLAS implements the BLAS (Basic Linear Algebra Subprograms) for single and
double precision, real and complex, with a CBLAS-compatible C interface and a
Fortran 77 interface.  It follows the approach of
[BLIS](https://github.com/flame/blis): all level 3 operations are built from
the same few building blocks — cache blocking, packing of matrix blocks, and a
small architecture-specific *micro kernel* that updates an MR x NR tile of the
result in registers.  Everything except the micro kernels (and a few optional
level 1 kernels) is portable C++, so a new CPU architecture needs one
well-defined kernel, not a new BLAS.

ulmBLAS started in 2014 as part of the lecture *Software Basics for High
Performance Computing* at Ulm University, and it is still meant to be read:
the code shows, step by step, how a matrix product reaches the performance
of the hardware.  It is also a library with a portable build, a regression
test suite running on several compilers and architectures, and documented,
reproducible benchmarks.

- **Complete level 3 BLAS** (gemm, symm, hemm, syrk, herk, syr2k, her2k,
  trmm, trsm) for all four data types, and levels 1 and 2 (136 of the 142
  routines of the CBLAS standard, see [Interfaces](#interfaces)); `dgetrf`
  and `dlaswp` from LAPACK.
- **Micro kernels**: portable reference kernels, SSE3 kernels for x86-64,
  and an AVX/FMA 6 x 8 DGEMM kernel derived from BLIS — 2.8 times faster
  than the SSE kernel and on par with BLIS on the tested CPU.
- **Header-only C++ core** (templates, C++11 or newer), built with GNU Make;
  besides the compilers only POSIX tools are needed, no other build system
  and no BLAS library.
- **Regression tests** for every routine, all matrix layouts and corner cases
  (NaN, Inf, zero dimensions, invalid arguments), seven kernel and blocking
  configurations, the Netlib BLAS test suites, and a registry of known issues.

> ulmBLAS is listed in
> [Meta Quest's third-party software notices](https://www.meta.com/en/legal/quest/third-party-notices/14/)
> ("ulmBLAS Copyright (C) 2014 Michael C. Lehn. Ulm University").

## Contents

- [Performance](#performance)
- [Architecture](#architecture)
- [Micro kernels and platforms](#micro-kernels-and-platforms)
- [Adding a micro kernel](#adding-a-micro-kernel)
- [Getting started](#getting-started)
- [Testing](#testing)
- [Benchmark archive](#benchmark-archive)
- [Interfaces](#interfaces)
- [License, provenance and acknowledgments](#license-provenance-and-acknowledgments)
- [Background and citation](#background-and-citation)

## Performance

DGEMM on one core of an Intel Core i5-12500 (Alder Lake, AVX2/FMA), single
threaded, compared with BLIS 2.0 and OpenBLAS 0.3.34 built from source for the
same machine:

![DGEMM performance versus matrix size](benchmark_results/2026-10-24-heim-avx-fma/plots/dgemm_square.svg)

| n = m = k ≥ 256 | AVX/FMA kernel relative to |
|---|---|
| ulmBLAS SSE kernel | 2.76 – 2.89 |
| BLIS 2.0 (same assembly kernel) | 0.99 – 1.02 |
| OpenBLAS 0.3.34 | 0.96 – 1.00 |

The AVX/FMA kernel reaches up to 59.7 GFLOP/s (OpenBLAS: 61.2 GFLOP/s); the
other level 3 routines reach 2.3 to 2.9 times the performance of the SSE
kernel.  For small matrices (n ≤ 128) BLIS and OpenBLAS are partly faster.

![Speedup of the AVX/FMA micro kernel over the SSE micro kernel](benchmark_results/2026-10-24-heim-avx-fma/plots/dgemm_speedup.svg)

These numbers hold for this CPU and the documented configurations only.  The
[benchmark campaign](benchmark_results/2026-10-24-heim-avx-fma/README.md)
documents hardware, compilers, library versions and build options, source
revisions and methodology, and contains the raw measurements and further
figures (thin matrices, syrk/symm/trmm/trsm, choice of block sizes).

## Architecture

All level 3 routines share the structure of the BLIS GEMM algorithm.  For
C ← β C + α A B the operands are partitioned into blocks that fit into the
caches, and the blocks are copied ("packed") into contiguous panels in the
order in which the micro kernel reads them:

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
L1 cache (B~ panel) and L2 cache (A~ panel).  Its performance determines the
performance of all level 3 routines.

| Layer | Directory | Responsibility |
|---|---|---|
| BLAS interfaces | `interfaces/blas/C`, `interfaces/blas/F77`, `interfaces/lapack/C` | argument checks, `xerbla`, row major as transposed column major |
| Level 3 algorithms | `ulmblas/level3/*.tcc` | cache blocking (MC, KC, NC), structure of symmetric, Hermitian and triangular matrices |
| Packing | `ulmblas/level3/pack` | copying blocks into MR and NR panels, zero padding, symmetric and triangular blocks |
| Macro kernels | `ulmblas/level3/mkernel` | loops over the MR x NR tiles; edge tiles through a buffer; tiles on the diagonal of triangular results |
| Micro kernels | `ulmblas/level3/ukernel/{ref,sse,avx}` | the MR x NR update — the architecture-specific part |
| Configuration | `ulmblas/config` | kernel selection, block sizes, compile-time checks |

The macro kernels work with *any* register block size: the edge tiles of a
block (mr < MR or nr < NR) are computed by the micro kernel into a local
buffer, and the diagonal tiles of syrk, syr2k, herk and her2k are found by
comparing row and column ranges.  MR and NR need not divide each other — the
AVX/FMA kernel uses MR = 6, NR = 8, and a test configuration runs the
reference kernel with these block sizes on every platform.  Static checks
enforce the remaining constraints (MC a multiple of MR and NR, NC a multiple
of NR, so that the packed panels fit the buffers).

## Micro kernels and platforms

| Kernel | Data types | MR x NR | Instruction set | Selected |
|---|---|---|---|---|
| reference (`ukernel/ref`) | all four | 4 x 4 (double; 8 x 4, 4 x 2, 2 x 2 for float, complex float, complex double) | portable C++ | without SSE3, and on all non-x86 targets |
| SSE3 (`ukernel/sse`) | double: GEMM and TRSM kernels; level 1 kernels | 4 x 4 | x86-64 SSE3, inline assembly | by default on x86-64 |
| AVX/FMA (`ukernel/avx`) | double: GEMM kernel | 6 x 8 | x86-64 AVX and FMA, inline assembly derived from BLIS | with `ARCHFLAGS='-mavx -mfma'` (or e.g. `-march=native`) |

Kernels are selected at compile time from the compiler's target macros
(`ulmblas/config/simd.h`); there is no runtime dispatch, so a library built
with AVX/FMA requires a CPU with AVX and FMA.  Types without an optimized
kernel use the reference kernel.

| Platform | Built | Test suite run | Optimized kernels executed and benchmarked |
|---|---|---|---|
| Linux x86-64, Debian 12, Intel Core i5-12500; GCC 12, clang 16 | yes | full `make check`, all seven configurations | SSE3 and AVX/FMA executed, both benchmarked |
| macOS arm64 (Apple silicon); Apple clang 21, GCC 16 | yes | full `make check` (the AVX/FMA configurations are skipped) | — (reference kernels) |
| macOS x86-64 under Rosetta 2; Apple clang 21 | yes | numeric tests (SSE3 kernels) | SSE3 executed, not benchmarked; AVX/FMA only compiled (Rosetta 2 provides no AVX) |
| 32-bit x86 | compiled (all interface sources; reference kernels), not linked | no | — |

Other platforms supported by a C++11 compiler and GNU Make should work with
the reference kernels but have not been tested.  Only x86-64 has optimized
kernels so far.  Details: [BUILDING.md](BUILDING.md#tested-configurations).

## Adding a micro kernel

A micro kernel for a new architecture or data type plugs into the existing
algorithms: all routines that use the GEMM micro kernel — the complete
level 3 BLAS — benefit without further changes.  Kernels from BLIS are a
natural source (BLIS has kernels for many x86-64, ARM, POWER and RISC-V
processors), and the AVX/FMA kernel shows the process.  A BLIS kernel cannot
simply be copied, though; an integration has to address:

- **Interface.**  ulmBLAS calls
  `ugemm(kc, alpha, A, B, beta, C, incRowC, incColC, nextA, nextB)` with
  references to alpha and beta and strides of C; BLIS kernels have a
  different signature and context arguments.
- **Packing layout.**  A is packed as MR x kc panels (column by column), B as
  kc x NR panels (row by row), as in BLIS; kernels expecting other layouts
  need adapted packing.
- **Block sizes.**  MR, NR and the cache block sizes MC, KC, NC are set per
  kernel and data type in `ulmblas/config/blocksize.h`; MC and KC must be
  chosen (measured) for the cache sizes of the target.
- **Edge tiles and strides.**  ulmBLAS calls the kernel for full tiles only;
  edge tiles go through a buffer.  If the assembly code supports only unit
  row or column strides of C (as the BLIS kernel), the kernel needs a fallback
  for general strides.
- **Numerical semantics.**  With β = 0, C must not be read (it may contain
  NaN); β = NaN must not be taken as zero.
- **Alignment and memory.**  The packing buffers are aligned for the kernel
  (32 bytes for AVX) and padded for reading ahead; inline assembly must declare
  all modified registers and memory.
- **Instruction set, compiler and object format.**  The kernel is selected by
  the compiler's target macros; inline assembly must assemble with GCC and
  clang and work on ELF and Mach-O (e.g. numeric local labels).
- **Tests.**  A test configuration with `CONFIG_REQUIRES` runs the complete
  test suite with the new kernel where the CPU supports it and reports SKIP
  elsewhere; `test/numeric/ukernel.cc` tests the micro kernel directly.
- **License and attribution.**  The upstream copyright and license notices
  must be preserved (see [License](#license-provenance-and-acknowledgments)).

The AVX/FMA kernel (`ulmblas/level3/ukernel/avx/ugemm.tcc`) lists its
provenance and every modification of the BLIS code.

## Getting started

Requirements: GNU Make 3.81 or newer and a C++11 compiler; the Netlib test
suites additionally need a Fortran and a C compiler.

```sh
git clone https://github.com/michael-lehn/ulmBLAS.git
cd ulmBLAS
make                              # libcblas.a, cblas.h, libclapack.a, clapack.h
make ARCHFLAGS='-mavx -mfma'      # the same with the AVX/FMA kernel (x86-64)
make -j8 check                    # build and run all tests
make clean                        # remove all generated files
make help                         # all targets and settings
```

A program using the CBLAS interface ([examples/dgemm.c](examples/dgemm.c)):

```c
#include <cblas.h>

/* column major: A is 2 x 3, B is 3 x 2, C is 2 x 2 */
double A[] = { 1, 4,  2, 5,  3, 6 };
double B[] = { 1, 0, 2,  0, 1, 1 };
double C[4];

cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 3,
            1.0, A, 2, B, 3, 0.0, C, 2);          /* C = [ 7 5 ; 16 11 ] */
```

```sh
cc -I. -c examples/dgemm.c -o dgemm.o
c++ -o dgemm dgemm.o libcblas.a                   # the library needs the C++ runtime
./dgemm
```

Compilers, flags and the cross compilation of the libraries are described in
[BUILDING.md](BUILDING.md).

## Testing

`make check` builds the libraries for seven configurations and runs, for
each of them:

- **C++ tests** of every routine of the C interface for the four data types:
  row and column major, all transposition, side, triangle and diagonal
  options, negative increments, padded leading dimensions, zero dimensions,
  alpha and beta of 0, 1 and other values, NaN where BLAS does not reference
  an operand, invalid arguments; results are compared with reference
  implementations using error bounds.
- **Micro kernel tests** of the selected and the reference kernel: full and
  edge tiles, all strides of C, k = 0, β = 0 with NaN and Inf in C, β = NaN,
  NaN and Inf in A and B.
- **Netlib BLAS test suites** for the C and the Fortran 77 interface.

The configurations combine tiny block sizes (so that small matrices cross
all block boundaries), the default block sizes, MR x NR of 4 x 4, 2 x 4,
4 x 2 and 6 x 8, and the reference, SSE3 and AVX/FMA kernels.  Configurations
whose CPU features are not available (e.g. AVX/FMA on ARM or under Rosetta 2)
are reported as SKIP with the reason; build errors, crashes and failures of
supported configurations are never skipped.

Every test has a stable ID.  Known bugs are XFAIL tests until they are
fixed, open design questions are DESIGN tests recording the current
behavior; both are listed in [test/known_issues.md](test/known_issues.md)
together with resolved issues and their regression tests.  `make check`
fails on FAIL and on XPASS (a known bug that no longer occurs).  Options and
the test layout: [BUILDING.md](BUILDING.md#tests).

## Benchmark archive

[benchmark_results/](benchmark_results/README.md) collects benchmark
campaigns over time: different processors, compilers, micro kernels and
blocking strategies.  Each campaign contains the raw measurements, CSV files,
Gnuplot scripts and the generated SVG figures, the benchmark programs, and
documentation of hardware, software, source revisions and methodology.

A completed campaign directory is added with

```sh
make add_benchmark_result BENCHMARK_DIR=path/to/campaign   # needs Gnuplot
```

which validates the campaign, regenerates its figures and updates the index;
it never overwrites archived measurements and does not run benchmarks.  The
ATLAS-based benchmark programs in `bench/` (`make bench`) are described in
[BUILDING.md](BUILDING.md#benchmarks).

## Interfaces

| Library | Header | Contents |
|---|---|---|
| `libcblas.a` | `cblas.h` | CBLAS (`cblas_dgemm`, ...): levels 1, 2 and 3 for float, double, complex float and complex double |
| `libclapack.a` | `clapack.h` | `ulmblas_dgetrf` (LU factorization with partial pivoting) and `ulmblas_dlaswp` |
| `libf77blas.a` | `f77blas.h` | Fortran 77 BLAS interface (`make f77blas`) |
| `libatlblas.a`, `libatllapack.a` | `atlblas.h`, `atllapack.h` | ATLAS names, used by the benchmarks |

Not implemented are `sdsdot`, `dsdot`, `srotm` and `srotmg`; `drotm` and
`drotmg` exist only as empty functions (for linking the ATLAS benchmarks).
The C++ templates in `ulmblas/` are the implementation, not a stable public
interface.

## License, provenance and acknowledgments

ulmBLAS is licensed under the BSD 3-Clause License (Copyright (c)
2014-2026, Michael C. Lehn and the ulmBLAS contributors), see
[LICENSE](LICENSE).  Third-party components keep their own copyright and
license terms, which [LICENSE](LICENSE) reproduces:

- **BLIS**: the design of ulmBLAS follows BLIS, and ulmBLAS uses code,
  concepts and ideas of the BLIS framework (BSD 3-Clause, The University of
  Texas).
- **AVX/FMA micro kernel**: derived from the micro kernel
  `bli_dgemm_haswell_asm_6x8` of BLIS 2.0 (Copyright (C) 2014, The
  University of Texas at Austin; Copyright (C) 2018 - 2019, Advanced Micro
  Devices, Inc.; BSD 3-Clause), as adapted for the course High Performance
  Computing I (Ulm University, 2025/26); provenance and modifications are
  listed in `ulmblas/level3/ukernel/avx/ugemm.tcc`.
- **ATLAS** timing programs and headers in `bench/` (ATLAS 3.10.1, BSD-style
  license; R. Clint Whaley, Antoine P. Petitet, Jeff Horner and others).
- **Reference BLAS** in `refblas/` (from LAPACK 3.3.1) and the **BLAS and
  CBLAS test programs** in `test/F77/` and `test/C/` (from Netlib), under
  the terms of Netlib and LAPACK.

The comparison benchmarks use BLIS and OpenBLAS, which are not part of
ulmBLAS.  ulmBLAS was written by Michael C. Lehn (Ulm University), with an
early contribution by Sebastian Kölle (GitHub: aquileia; repository setup,
plots of the benchmarks).

## Background and citation

The design of ulmBLAS was strongly inspired by

Field G. Van Zee and Robert A. van de Geijn. *BLIS: A Framework for Rapidly
Instantiating BLAS Functionality.* ACM Transactions on Mathematical Software,
41(3), 14:1–14:33, June 2015.
[https://doi.org/10.1145/2764454](https://doi.org/10.1145/2764454)
(at the time of the initial development, mainly in 2014, the
[preprint](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf)).

ulmBLAS is part of the lecture *Software Basics for High Performance
Computing* (MATH9367, Ulm University):
[course page for ulmBLAS](http://www.mathematik.uni-ulm.de/~lehn/ulmBLAS),
[general course page](http://www.mathematik.uni-ulm.de/~lehn/sghpc), and
step by step from a naive loop to peak performance:
[Achieving peak performance for GEMM](http://www.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html).
A related project is [ulmBLAS-core](https://github.com/michael-lehn/ulmBLAS-core).

If you use ulmBLAS in your work, please cite it
([CITATION.cff](CITATION.cff)):

Michael C. Lehn. *ulmBLAS: A high performance BLAS implementation.* Ulm
University, GitHub. https://github.com/michael-lehn/ulmBLAS, 2014.

```bibtex
@misc{Lehn_ulmBLAS,
  author       = {Michael C. Lehn},
  title        = {ulmBLAS: A high performance BLAS implementation},
  year         = {2014},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/michael-lehn/ulmBLAS}},
  note         = {Ulm University, Version 1.0}
}
```
