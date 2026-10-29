# GEMM benchmark: ulmBLAS and the BLAS libraries on your machine

`make bench-gemm` measures single-threaded GEMM (C ← A·B + C, square
m = n = k, column major) of ulmBLAS and of every other BLAS library it
finds, and writes the raw results, a summary and a plot.  No other library
is needed to benchmark ulmBLAS itself.

```sh
make bench-gemm                              # DGEMM, ulmBLAS + all libraries found, n = 50 ... 4000
make bench-gemm PREC=z                       # ZGEMM
make bench-gemm LIBS="ulmblas openblas"      # only these (each must be found)
make bench-gemm SIZES="100 200 400" SAMPLES=1   # a quick check
```

## What is measured

| | |
|---|---|
| Operation | `cblas_?gemm(ColMajor, NoTrans, NoTrans, n, n, n, 1, A, n, B, n, 1, C, n)`; for Eigen the C++ expression `C.noalias() += A*B` (Eigen has no CBLAS interface) |
| Data types | `PREC=s` SGEMM, `d` DGEMM (default), `c` CGEMM, `z` ZGEMM |
| Sizes | m = n = k = 50, 100, 150, …, 4000 (80 sizes), `SIZES` overrides |
| GFLOP/s | 2·m·n·k floating-point operations for SGEMM, DGEMM; 8·m·n·k for CGEMM, ZGEMM (a complex multiply-add is 4 real multiplications and 4 additions) |
| Timing | per size: one call, checked; a warm-up of at least 0.3 s; then `SAMPLES` (default 3) samples of at least 0.2 s each, the median is reported; operands stay in place (hot caches) |
| Check | 400 random elements of C of the first call are compared with sums in long double (error bound 4·(k+2)·ε·Σ\|terms\|); a failed check is reported as `WRONG` |
| Threads | one: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`, `MKL_NUM_THREADS` = 1, `MKL_DYNAMIC=FALSE`, `VECLIB_MAXIMUM_THREADS` = 1; oneMKL is linked sequential; Eigen is built without OpenMP |
| Rounds | `ROUNDS` (default 1) passes over all libraries, the order rotated per round; the summary uses the median over the rounds |
| Pinning | `CPU=3` runs every program on CPU 3 with `taskset` (Linux); macOS has no pinning |

The source is [benchx.c](benchx.c) (CBLAS libraries) and
[bench-eigen.cpp](bench-eigen.cpp) (Eigen, same timing and check); the
driver is [run.sh](run.sh).

## Libraries

| Name in `LIBS` | Library | Found | Override |
|---|---|---|---|
| `ulmblas` | ulmBLAS of this tree (`libcblas.a`, built by `make bench-gemm` with `ARCHFLAGS`) | always | `ARCHFLAGS` select the micro kernels |
| `openblas` | OpenBLAS | automatically: pkg-config `openblas`, Homebrew (`brew --prefix openblas`), or `-lopenblas` on the default paths | `OPENBLAS_DIR` (with `include/`, `lib/`) |
| `blis` | BLIS | automatically: pkg-config `blis`, or `-lblis` with `/usr/include/blis` | `BLIS_DIR` (with `include/blis/`, `lib/`) |
| `aocl` | AOCL-BLAS (AMD's BLIS) | only with a directory: its library is also called `libblis` | `AOCL_DIR` or `AOCL_ROOT` |
| `mkl` | Intel oneMKL (sequential, LP64) | when `MKLROOT` is set (e.g. by oneAPI's `setvars.sh`) | `MKLROOT` |
| `accelerate` | Apple Accelerate | automatically on macOS | — |
| `eigen` | Eigen (header only, its own kernels) | automatically: pkg-config `eigen3` | `EIGEN_DIR` (directory containing `Eigen/`), `EIGEN_FLAGS` (default `-march=native`, on ARM `-mcpu=native`) |

Not supported by `make bench-gemm`: ATLAS and the Netlib reference BLAS.
They (and oneMKL) can be compared with the older ATLAS timing programs in
[`bench/`](../) (`make bench`: double precision only, libraries configured
explicitly).

**Found means verified.**  For every candidate, `run.sh` builds the
benchmark program, runs it once and requires that it reports the expected
library: `openblas_get_config()` for OpenBLAS, `bli_info_get_version_str()`
for BLIS and AOCL-BLAS (a library that reports AOCL-BLAS is not accepted as
BLIS), `mkl_get_version_string()` for oneMKL, the Accelerate framework as
the object that contains `cblas_?gemm`.  Each program also prints the file
its `cblas_?gemm` comes from.  A library that is not found is skipped with
the reason (`LIBS=auto`, the default); a library named explicitly in `LIBS`
must be found, otherwise the run stops.  One library is never used in
place of another.

## Results

`OUT` (default `bench/gemm/results/YYYYMMDD-HHMMSS-?gemm`) contains

| File | Contents |
|---|---|
| `detection.txt` | machine, compilers, ulmBLAS `ARCHFLAGS`, threads, sizes; for every library: selected or skipped (and why), where it was found, what the program reported; the kernels and block sizes ulmBLAS selected for each type |
| `build.log` | the compile and link command of every program and its messages |
| `raw/LIB-ROUND.txt` | the measurements: `m n k median min max calls_per_sample max_error/bound check`, after `#` lines identifying the library |
| `summary.md` | GFLOP/s at a few sizes, the median ratio to ulmBLAS, failed checks |
| `gflops.dat`, `gflops.gp`, `gflops.svg` | GFLOP/s over n for all libraries; the plot if Gnuplot is installed |

## ulmBLAS micro kernels

The micro kernels of ulmBLAS are selected at compile time by `ARCHFLAGS`
(see the [micro kernel README](../../ulmblas/level3/ukernel/README.md));
`make bench-gemm` rebuilds `libcblas.a` when they change, and
`detection.txt` lists the kernel set, MR x NR and MC, KC, NC per type:

```sh
make bench-gemm ARCHFLAGS='-mavx -mfma'                    # x86-64: AVX2/FMA kernel
make bench-gemm ARCHFLAGS='-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12'
make bench-gemm ARCHFLAGS=-mcpu=apple-m4                   # Apple M4/M5: SME2 kernel (clang)
```

One run benchmarks one ulmBLAS build; compare several kernels with several
runs (different `OUT`).

## Interpreting the numbers

The results depend on the CPU and its frequency behavior, the compiler, how
each library was built (e.g. a distribution's generic OpenBLAS build with
run-time dispatch versus one built for the CPU), the kernel each library
selects at run time, and the threading configuration; `detection.txt`
records what could be determined.  Only one core is used; the libraries
differ much more in their multi-threaded behavior.  Measure on an otherwise
idle machine.  The documented campaigns in
[`benchmark_results/`](../../benchmark_results/README.md) follow the same
method with more control (pinning, load gate, interleaved rounds).
