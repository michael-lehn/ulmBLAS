# ulmBLAS

**A high performance BLAS implementation**

---
**Notable use:** ulmBLAS is listed in
[Meta Quest's official third-party software notices](https://www.meta.com/en/legal/quest/third-party-notices/14/)
as
> **ulmBLAS Copyright (C) 2014 Michael C. Lehn. Ulm University**
---

This library is part of my lecture:

**Software Basics for High Performance Computing** (MATH9367)  
Ulm University

- [Course page for ulmBLAS](http://www.mathematik.uni-ulm.de/~lehn/ulmBLAS)
- [General course page](http://www.mathematik.uni-ulm.de/~lehn/sghpc)

---

And yes, I am particularly proud of the section demonstrating how to achieve
peak performance for the matrix-matrix product:

👉 [Achieving peak performance for GEMM](http://www.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html)

---

**Note:** Further development will take place in
[ulmBLAS-core](https://github.com/michael-lehn/ulmBLAS-core).

---
## Building and testing

ulmBLAS is built with **GNU Make 3.81 or newer** (no other build system is
needed or supported).

```sh
make              # libcblas.a, cblas.h, libclapack.a, clapack.h
make -j8 check    # build and run the tests
make help         # all targets and the current settings
make clean        # remove all generated files
```

### GNU Make

- The minimum version is 3.81 (the build uses `$(abspath)` and `$(lastword)`).
  Older versions stop with an error message naming the detected and the
  required version.
- On macOS `/usr/bin/make` is GNU Make 3.81, which works.  A newer GNU Make
  from Homebrew or MacPorts is installed as `gmake`.  Sub-makes are always
  invoked through `$(MAKE)`, so `gmake` works as well.
- On the BSDs GNU Make is installed as `gmake`.  BSD make reads `Makefile`,
  which only explains that GNU Make is required (the build itself is in
  `GNUmakefile`, which GNU Make reads first).

### Requirements

| Tool | Needed for | Variable |
| --- | --- | --- |
| C++ compiler, C++11 or newer | all libraries, C++ tests | `CXX` (default `c++`), `CXXSTD` (default `-std=c++11`) |
| `ar`, `ranlib` | static libraries | `AR`, `ARFLAGS`, `RANLIB` |
| Fortran 77 compiler (gfortran) | `make f77blas`, `make refblas`, Netlib test suites, benchmarks | `FC` (default `gfortran`), `FLIBS` |
| C compiler | Netlib CBLAS test suite, header tests, benchmarks | `CC` (default `cc`) |
| gnuplot | plots of the benchmarks (optional) | `GNUPLOT` |

The default target and `make check-numeric` only need a C++ compiler and
`ar`.  No reference BLAS or LAPACK library is needed: the tests bring their
own reference implementations.  No scripting language besides the POSIX
shell and awk is used.

Before building, each tool a target needs is checked by compiling (and,
for the tests and benchmarks, linking) a tiny program (`mk/probe.sh`).  If a
tool is missing or not usable, the error explains what it is needed for and
how to select another one.  Fortran objects are linked with the C++
compiler; for gfortran the Fortran runtime (`FLIBS`) is located
automatically, for other Fortran compilers set `FLIBS` explicitly.

Building the libraries never executes programs for the target, so the
libraries can be cross compiled (e.g. `make CXX="clang++ -arch x86_64"` on
an ARM Mac).  `make check` and `make bench` run the programs they build and
therefore need a native toolchain.

### Selecting compilers and flags

All settings can be given on the command line or in the environment (see
`make.inc`):

```sh
make CXX=g++-16 CC=gcc-16 FC=gfortran-16
make CXX=clang++ CXXSTD=-std=c++20
make CXXFLAGS=-g LDFLAGS=... LDLIBS=...
make OPTFLAGS=-O0 CXXFLAGS=-g          # debug build without optimization
```

`CPPFLAGS`, `CXXFLAGS`, `CFLAGS`, `FFLAGS`, `LDFLAGS` and `LDLIBS` are user
variables: setting them never removes flags required by the build (include
paths, interface macros, test block sizes), which are kept in separate
variables.  The optimization flags are kept separately as well:
`OPTFLAGS = -O3 -fomit-frame-pointer` (C++), `FOPTFLAGS = -O3` (Fortran)
and `COPTFLAGS = -O2` (C, test drivers and benchmark programs only).  They
precede `CXXFLAGS`, `FFLAGS` and `CFLAGS` on the command lines, so `CXXFLAGS`
set in the environment (e.g. for include paths) does not disable the
optimization, while an `-O` option in `CXXFLAGS` takes precedence (the last
`-O` option wins).  To build without optimization use `OPTFLAGS=-O0`.  The
defaults of the user variables are `CXXFLAGS = -Wall` and
`FFLAGS = -Wall -fimplicit-none`.  As usual for GNU Make, variables given on
the command line take precedence over the environment.  `ARCHFLAGS` defaults to
`-msse3 -mfpmath=sse` if the C++ compiler accepts these flags (x86), which
selects the optimized SSE kernels; on other architectures (e.g. ARM) the
portable reference kernels are used.  With `ARCHFLAGS='-mavx -mfma'` (or
e.g. `-march=native` on a CPU with AVX and FMA) the AVX/FMA micro kernel for
double (MR = 6, NR = 8, derived from BLIS, see `LICENSE`) replaces the SSE
micro kernel; the libraries then require a CPU with AVX and FMA.  Objects are rebuilt when the compiler
or the flags change, and header dependencies are tracked.

### Libraries

| Target | Files (in the top-level directory) |
| --- | --- |
| `cblas` | `libcblas.a`, `cblas.h` (CBLAS, functions `cblas_*`) |
| `clapack` | `libclapack.a`, `clapack.h` (`ulmblas_dgetrf`, `ulmblas_dlaswp`) |
| `atlblas`, `atllapack` | `libatlblas.a`, `atlblas.h`, `libatllapack.a`, `atllapack.h` (ATLAS names, used by the benchmarks) |
| `f77blas` | `libf77blas.a`, `f77blas.h` (Fortran 77 interface) |
| `refblas` | `librefblas.a` (Netlib reference BLAS, for benchmarks) |

### Tests

```sh
make check                    # all tests
make check-numeric            # only the C++ tests (no Fortran needed)
make check-netlib             # only the Netlib BLAS test suites
make check CONFIGS=testcase04 # only the default configuration (faster)
make check TESTS=gemm         # only tests whose ID or name contains "gemm"
make check VERBOSE=1          # print every result
make check STRICT=1           # skipped tests are errors
```

All tests are run for seven configurations (`test/config`): four with tiny
block sizes and the reference kernels (so that small matrices cover multiple
and partial blocks of the GEMM algorithm; one of them with MR = 6 and NR = 8,
which do not divide each other), one with the default block sizes and
kernels, and two with the AVX/FMA micro kernel (`-mavx -mfma`, default and
tiny block sizes).  The AVX/FMA configurations are only built and run if the
compiler, the CPU and the operating system support AVX and FMA; otherwise
their tests are reported as SKIP.  The libraries of each configuration are
built in `test/build/`.

- **C++ tests** (`test/numeric`): levels 1, 2 and 3 of the C interface for
  float, double, complex float and complex double, row and column major,
  all transpose/side/upper-lower/diagonal options, negative and non-unit
  increments, leading dimensions larger than needed, zero dimensions,
  alpha and beta of 0, 1 and general values (with NaN where BLAS documents
  that an operand is not referenced or need not be set, e.g. C for beta = 0
  or unit diagonals), matrices larger than the default block sizes MC and
  KC (the default NC = 4096 is only exceeded with the small test block
  sizes), and `dgetrf` (including pivot indices and row major storage) and
  `dlaswp`.  Invalid arguments are tested as well: each function of the C
  interface that checks its arguments must report an invalid argument
  through `cblas_xerbla` and return without touching its operands.  Results are compared with straightforward reference
  implementations using error bounds, not bitwise.  Padding and elements
  that must not be written are checked bitwise.
- **Netlib BLAS test suites** (`test/F77`, `test/C`): the official test
  programs for the Fortran 77 and the C interface.  They need a Fortran and
  a C compiler.  If these are not usable, `make check` reports the suites
  as SKIP with the reason (and fails with `STRICT=1`).
- **Header tests**: the generated headers can be used from C.

Every test has a stable ID.  Results are PASS, FAIL, XFAIL (known bug),
XPASS (known bug no longer observed: an error), DESIGN (open design
question, observed behavior as recorded) and SKIP.  Known bugs and open
design questions are listed in [`test/known_issues.md`](test/known_issues.md).
`make check` exits with a nonzero status if a test fails.

### Benchmarks

```sh
make bench                    # long running, not part of make check
make bench USE_BLAS_VARIANTS="ulmBLAS netlib" USE_BENCH_CASES=gemm
```

The benchmarks (ATLAS timing programs in `bench/`) need the C, C++ and
Fortran compilers.  `make bench` first builds all libraries the benchmark
programs are linked with (each library once), then the benchmark programs;
`make benchmark-suite` only builds the programs.  A benchmark is run again
when its program (or a library it is linked with) or its configuration has
changed.  Results are written to `bench/benchmarks/`; plots are created if
gnuplot is found.  Comparisons with ATLAS or Intel MKL need `ATLAS_LIBDIR`
or `MKLROOT` (variants `atlas`, `mkl`).

### Benchmark results

Published benchmark campaigns (measurements, Gnuplot scripts and SVG
figures, documentation of hardware, software and methodology) are archived
in [`benchmark_results/`](benchmark_results/README.md), e.g. the
[AVX/FMA DGEMM micro kernel on an Intel Core i5-12500](benchmark_results/2026-10-24-heim-avx-fma/README.md).
A completed campaign is added with

```sh
make add_benchmark_result BENCHMARK_DIR=path/to/campaign   # needs gnuplot
```

which validates it, regenerates its figures and updates the index (see
`benchmark_results/README.md`).

### Tested configurations

Tested (October 2026) on macOS 26 (arm64) with GNU Make 4.4.1 and 3.81:

- full `make check` with Apple clang 21 (`c++`) and GCC 16.2 (`g++-16`,
  `gcc-16`), both with gfortran 16.2;
- `make check-numeric` with GCC 16.2 and Apple clang 21 (C++17, C++20,
  C++23), LLVM clang 17 (C++11) and 23 (C++23), and with AddressSanitizer
  and UndefinedBehaviorSanitizer;
- compilation of the libraries only: LLVM clang 18 to 22 (C++11), GCC 16.2
  and Apple clang 21 with C++14 and C++26 (`-std=c++2c`);
- cross compilation for x86-64 with Apple clang 21
  (`CXX="clang++ -arch x86_64"`, SSE kernels selected): `make check-numeric`
  run under Rosetta 2 (the Netlib tests need an x86-64 gfortran, which was
  not available).

Tested (October 2026) on Debian 12 (x86-64, Intel Core i5-12500) with GNU
Make 4.3: full `make check` with GCC 12.2 (`g++`, `gcc`) and with clang 16
(`clang++-16`, `clang-16`), both with gfortran 12.2; the SSE kernels are
selected by default, the AVX/FMA micro kernel is tested by the
configurations 06 and 07 (and with `ARCHFLAGS='-mavx -mfma'`).

C++11 is the minimum (C++98 is rejected: the code uses `decltype`, `auto`
and `<type_traits>`).

On arm64 the portable reference kernels are used.  The x86 SSE kernels
were tested on native x86-64 hardware (Linux, GCC 12 and clang 16) and on
macOS x86-64 under Rosetta 2 (Apple clang 21), not on native macOS x86-64
hardware.  The AVX/FMA micro kernel was tested on Linux x86-64 (GCC 12,
clang 16); for macOS x86-64 it was only compiled (Rosetta 2 does not
provide AVX).  BSD was not tested.  The
macOS system GNU Make 3.81 was tested; GNU Make 3.80 and older are rejected
by a version check, which was only tested by simulating the version.

---
## Background

The design and implementation of ulmBLAS was strongly inspired by the ideas
presented in:

Field G. Van Zee and Robert A. van de Geijn. *BLIS: A Framework for Rapidly
Instantiating BLAS Functionality.* ACM Transactions on Mathematical Software,
41(3), 14:1–14:33, June 2015.
[https://doi.org/10.1145/2764454](https://doi.org/10.1145/2764454)

At the time of the initial development of ulmBLAS (mainly in 2014), I was
referring to a preprint version of this paper:
[https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf)

---


## Citation

ulmBLAS is an independent project developed and maintained by **Michael C. Lehn** (Ulm University).  
If you use ulmBLAS in your work, please cite it as:

Michael C. Lehn. *ulmBLAS: A high performance BLAS implementation.* Ulm University, GitHub. https://github.com/michael-lehn/ulmBLAS, 2014.

### BibTeX:

```bibtex
@misc{Lehn_ulmBLAS,
  author       = {Michael C. Lehn},
  title        = {ulmBLAS: A high performance BLAS implementation},
  year         = {2014},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/michael-lehn/ulmBLAS}},
  note         = {Ulm University, Version 1.0}
}

