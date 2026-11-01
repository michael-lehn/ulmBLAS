# Installing ulmBLAS

The short version is in the [README](README.md#how-to-install).  This
file has the details: requirements, choosing the micro kernels with
`make check_arch` and `ARCHFLAGS`, tests, and installing the libraries
and headers.  Compilers, flags, cross compilation and the tested
configurations: [BUILDING.md](BUILDING.md).

## Requirements

**Required:** GNU Make 3.81 or newer (on macOS the system `make` works; on
the BSDs use `gmake`) and a C++11 compiler (`c++`, `g++`, `clang++`); the
build needs no other tools and no BLAS library.  **Optional:** a C and a
Fortran compiler (`cc`, `gfortran`) for the Fortran 77 interface and the
Netlib test suites of `make check`; Python 3 for the benchmark summary and
the archive checks; Gnuplot for plots.  Details: [BUILDING.md](BUILDING.md).

## Build

Build the C interface of BLAS (`libcblas.a`, `cblas.h`) and of the
LAPACK routines (`libclapack.a`, `clapack.h`) in the top-level directory:

```sh
git clone https://github.com/michael-lehn/ulmBLAS.git
cd ulmBLAS
make check_arch                   # supported ARCHFLAGS, with a suggestion
make                              # SSE3 kernels on x86-64, otherwise reference kernels
make help                         # all targets and settings
```

## Choosing the micro kernels

The micro kernels are selected at compile time from the target of the
compiler: on x86-64 with SSE3 the SSE kernels for double (`make.inc` adds
`-msse3` to the default `ARCHFLAGS` if the compiler accepts it), everywhere
else and for the other data types the portable reference kernels.  There is
nothing to choose on this branch; `make check_arch` shows what is selected:

```sh
make check_arch
```

Wider instruction sets (e.g. `ARCHFLAGS=-march=native`) do not select a
faster kernel.  `make f77blas` builds the Fortran 77 interface
(`libf77blas.a`, `f77blas.h`; needs `FC`).

## Index type

Dimensions, increments, leading dimensions and indices (pivots, the
result of `i?amax`, `info` of `dgetrf`) of the C, Fortran 77 and LAPACK
interfaces have the type `ULMBLAS_INDEX_TYPE`, `int` by default, as in the
reference BLAS.  Like MKL's `MKL_INT` it can be a 64-bit type, chosen when
the libraries are built:

```sh
make ULMBLAS_INDEX_TYPE=int64_t           # or long, "long long"
make CPPFLAGS=-DULMBLAS_INDEX_TYPE=int64_t   # the same
```

The generated headers (`cblas.h`, `clapack.h`, `f77blas.h`) define the
macro `ULMBLAS_INDEX_TYPE` with the type of the build and declare every
function with it, so a program compiled against these headers uses the
type of the library.  A program that defines the macro itself must use the
same type (for a different one the compiler warns that the macro is
redefined, and the header's definition applies).  The
error handler `cblas_xerbla` keeps its `int` argument (a parameter number).

- **Fortran 77 interface**: Fortran passes `INTEGER` arguments, so with a
  64-bit index type the Fortran code (the wrappers of `libf77blas.a` and
  the calling program) must be compiled with 64-bit `INTEGER`, e.g.
  `make f77blas ULMBLAS_INDEX_TYPE=int64_t FFLAGS=-fdefault-integer-8` with
  gfortran.
- **Inside**, independently of `ULMBLAS_INDEX_TYPE`, the interfaces call
  the C++ templates with `IndexType = std::ptrdiff_t`, so offsets such as
  `j*ldC` never overflow, even if all arguments fit into an `int`.
- **Tests**: `make check` runs the configurations `testcase10`
  (`int64_t`) and `testcase11` (`long long`) besides the `int`
  configurations; the Netlib CBLAS test programs, which pass `int`, are
  reported as SKIP there, the Fortran 77 test programs run with 64-bit
  `INTEGER`.

## Shared library for Julia

```sh
make lbt                          # libulmblas_lbt.so (macOS: .dylib)
```

- For [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline),
  through which Julia calls BLAS: the Fortran 77 interface with 64-bit
  integers (ILP64) and the symbol suffix `_64_` (`dgemm_64_`), as in
  OpenBLAS's ILP64 build.
- No Fortran compiler needed: the Fortran functions (`ddot`, ...) are C
  code in this library ([interfaces/blas/lbt/](interfaces/blas/lbt/)).
- The library also contains the GEMM with arbitrary strides
  (`ulmblas_?gemm_strided`,
  [ulmblas_strided.h](interfaces/blas/C/ulmblas_strided.h)), for calls
  with `ccall`; libblastrampoline does not know these names and ignores
  them.
- On Linux the C++ runtime is linked statically, so the library does not
  depend on the `libstdc++` of the compiler (Julia brings its own).
- `OUTDIR` and `BUILD_DIR` keep several builds apart, e.g. one per
  instruction set.

In Julia (LAPACK and all routines ulmBLAS does not have stay with the
library Julia uses by default, usually OpenBLAS):

```julia
using LinearAlgebra
BLAS.lbt_forward("/path/to/libulmblas_lbt.so"; clear=false)
BLAS.get_config()                 # lists libulmblas_lbt
```

## Test

Test the libraries (`make` rebuilds them for every test configuration
in `test/build/`):

```sh
make -j8 check                    # all tests (needs C and Fortran compilers)
make -j8 check-numeric            # only the C++ tests: no Fortran needed
```

Without a usable Fortran compiler, C compiler or Fortran runtime the
Netlib suites are reported as SKIP with the reason (`STRICT=1` turns
skipped tests into errors).  `CONFIGS=...` selects configurations,
`TESTS=...` tests by ID or name, `VERBOSE=1` prints every result; all
results are written to `test/build/check.log`.

`make check` builds the libraries for every test configuration in
`test/config/` (ten by default) and runs, for each of them:

- **C++ tests** of every routine of the C interface for the four data types:
  row and column major, all transposition, side, triangle and diagonal
  options, negative increments, padded leading dimensions, zero dimensions,
  alpha and beta of 0, 1 and other values, NaN where BLAS does not reference
  an operand, invalid arguments; results are compared with reference
  implementations using error bounds.
- **Micro kernel tests** of the selected and the reference kernel: full and
  edge tiles, all strides of C, k = 0, β = 0 with NaN and Inf in C, β = NaN,
  NaN and Inf in A and B.
- **Netlib BLAS test suites** for the C and the Fortran 77 interface
  (`sblat1`, `dblat1` of LAPACK 3.12.1, which also test `?rotm`, `?rotmg`,
  `sdsdot`, `dsdot` and `nrm2` with Inf and NaN), and `ublat1` for the
  Fortran 77 interface: complex Givens rotations, `?cabs1` and the
  arguments passed to a user-supplied `XERBLA`.
- **Header tests**: `cblas.h` and `clapack.h` compile as C.
- **Examples**: the C++ examples of [examples/](examples) are compiled with
  the flags of the configuration (its kernels and block sizes) and run.

`make check-reference-lapack` (not part of `make check`, needs network
access) downloads Reference-LAPACK 3.12.1 (checksum verified) and runs its
unmodified BLAS and CBLAS test programs against ulmBLAS.

The configurations combine tiny block sizes (so that small matrices cross
all block boundaries), the default block sizes, MR x NR of 4 x 4, 2 x 4,
4 x 2, 6 x 8 and 32 x 16, and the reference and SSE3 kernels.  Configurations
whose CPU features are not available (the SSE3 and AVX/FMA configurations on
ARM) are reported as SKIP with the reason; build errors, crashes and failures
of supported configurations are never skipped.

Every test has a stable ID.  Known bugs are XFAIL tests until they are
fixed, open design questions are DESIGN tests recording the current
behavior; both are listed in [test/known_issues.md](test/known_issues.md)
together with resolved issues and their regression tests.  `make check`
fails on FAIL and on XPASS (a known bug that no longer occurs).  Options and
the test layout: [BUILDING.md](BUILDING.md#tests).

## Use

Programs use the CBLAS interface (`cblas.h`, `libcblas.a`, linked by a
C++ compiler) or the header-only C++ interface; examples:
[README: How to Use](README.md#how-to-use).

## Install

There is no `make install`: ulmBLAS consists of static
libraries and headers (and the shared library for Julia, see above), which
can be used from the source directory (as above) or copied to a directory
of your choice.  Choose one that does not
contain another BLAS: `cblas.h` and `libcblas.a` are also the names other
BLAS libraries use.

```sh
mkdir -p $HOME/opt/ulmblas/include $HOME/opt/ulmblas/lib
cp cblas.h clapack.h $HOME/opt/ulmblas/include/
cp libcblas.a libclapack.a $HOME/opt/ulmblas/lib/
cc -I$HOME/opt/ulmblas/include -c examples/dgemm.c -o dgemm.o
c++ -o dgemm dgemm.o -L$HOME/opt/ulmblas/lib -lcblas
```

For the header-only C++ interface, copy the directory `ulmblas/` (it
needs no library) and compile with `-I` pointing to its parent directory.

`make clean` removes all generated files.  Compilers, flags and cross
compilation: [BUILDING.md](BUILDING.md).
