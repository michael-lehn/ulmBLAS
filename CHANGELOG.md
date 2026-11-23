# Changelog

User-visible changes of ulmBLAS per release.  Versions: `vX.Y.Z` tags of
the Git repository (procedure: [RELEASING.md](https://github.com/michael-lehn/ulmBLAS/blob/master/RELEASING.md)).

## 0.4.3 (unreleased)

### Fixes

- macOS: `libulmlapack` re-exports the LAPACK (Linear Algebra PACKage)
  functions that `libulmblas` has had since 0.4.2 (`?getrf`, `?gesv`,
  `?potrf`, ...).  macOS binds every symbol to the library it came from
  when a program is linked, so programs linked against `libulmlapack`
  0.4.1 look for them there; with 0.4.2 they did not load (e.g. NumPy
  built with `interfaces/python/numpy.sh`: `Symbol not found: _cgesv_`).

## 0.4.2 (2026-11-23)

### New functions

- Error handlers set at run time, with the interface of the reference
  implementation of LAPACK (Linear Algebra PACKage), pull request #1407
  of Reference-LAPACK/lapack: `cblas_set_xerbla`, `cblas_get_xerbla`
  (the handler gets the position of the invalid parameter in the call of
  the C function and a `va_list`; the second argument of
  `cblas_set_xerbla`, the handler for the Fortran 77 interface, is not
  used: the C interface of ulmBLAS does not call it), `set_blas_xerbla_`,
  `get_blas_xerbla_`, and the ILP64 (64-bit integer) names with the
  suffixes `_64` and `_64_`.  A program that defines `cblas_xerbla` or
  `xerbla_` still replaces them.
- Hooks of FlexiBLAS in the shared library (as a FlexiBLAS backend):
  `flexiblas_backend_set_num_threads`, so that `flexiblas_set_num_threads`
  and its aliases (`openblas_set_num_threads`, ...) set the threads of
  ulmBLAS; `flexiblas_backend_get_num_threads`; `__flexiblas_set_xerbla`,
  through which FlexiBLAS passes the `XERBLA` of the program.
- The shared library `libulmblas` also exports the LAPACK functions of
  ulmBLAS (LP64 (32-bit integer) and ILP64 names): `?getrf`, `?getrs`,
  `?gesv`, `?getri`, `?potrf`, `?potrs`, `?posv`, `?potri`, `?trtrs`,
  `?trtri`, `?lauum` and `?laswp` for `s`, `d`, `c`, `z`.  `libulmlapack`
  (Reference-LAPACK) uses them instead of its own, and so does FlexiBLAS
  with the backend ulmBLAS (the rest of LAPACK from its reference
  implementation).
- `libulmblas64`: the interfaces with 64-bit integers under the usual
  names (`dgemm_`, `cblas_dgemm`, `xerbla_`, `lsame_` with a 64-bit
  `LOGICAL`, the LAPACK functions of ulmBLAS), as Debian's `libblas64`,
  for programs compiled with 64-bit `INTEGER` and as the backend of
  FlexiBLAS with 64-bit integers (`libflexiblas64` looks up only these
  names).  `make install-shared` installs it with
  `include/ulmblas64/cblas.h` and `ulmblas64.pc` (not on Windows); on
  Apple Silicon and x86-64 Linux it is a second front of the backends of
  `libulmblas`.

### Changed

- The parameter number of `cblas_xerbla` has the index type
  `ULMBLAS_INDEX_TYPE` (before: `int`), as `CBLAS_INT` of the reference:
  `cblas_xerbla_64` takes `int64_t`.  Unchanged for the default index type
  `int`.
- `libulmlapack` (Reference-LAPACK on top of ulmBLAS) is built with the
  makefiles of LAPACK instead of CMake: `make install-shared` needs no
  CMake any more.  Optimization as before (`-O2`, `LAPACK_FOPT`).
- The libraries link no C++ runtime (no `libstdc++`, no `libc++`; before:
  `libc++` on macOS, `libstdc++` statically on Windows and on Linux if
  `libstdc++.a` exists).  The library code is compiled without
  exceptions (`-fno-exceptions`) and the shared libraries are linked with
  `-nostdlib++` (clang, GCC 13 or newer); before linking, the objects are
  checked for references to the runtime.  `ULMBLAS_STATIC_LIBSTDCXX` only
  matters for GCC before 13.  Programs link `libcblas.a` with a C
  compiler (`cc ... libcblas.a -lm -pthread`); before, they needed a C++
  compiler.
- The build is one make: no make calls another (before: a makefile per
  directory, called with `$(MAKE) -C` for every configuration, backend and
  index type).  The rules of the libraries are generated for each build
  with `$(eval)` (`mk/`, see `mk/README.md`), so `make -j` builds
  everything with one job server and rebuilds only what is out of date.
  The targets and variables are the same; new targets `build-shared` (the
  build step of a package), `check-package` (its test) and `shared64`.
  `make install-shared` installs with make; `interfaces/install-shared.sh`
  only runs it, for the recipes that call the script.  `make sonames`
  (`ULMBLAS_SONAMES=yes`) also on Linux without the dispatching library
  (aarch64: the library linked as `libblas.so.3` and `libcblas.so.3`).

### Documentation

- Guides in `doc/guides/` (PDF with the example programs; not in the
  release tarball), written for the course High Performance Computing I
  at Ulm University: GEMM (general matrix-matrix product) and the
  level-3 BLAS (Basic Linear Algebra Subprograms), the LU (lower-upper)
  and Cholesky factorizations, vectorization and multithreading with
  OpenMP (Open Multi-Processing) and C++ atomics, and the C++ features
  ulmBLAS uses.  Overview: `doc/README.md`.

### Libraries and packaging

- Fedora: `flexiblas-ulmblas` also registers `libulmblas64` for FlexiBLAS
  with 64-bit integers (`flexiblas64rc.d`).
- openSUSE: the alternatives `libblas.so.3` and `libcblas.so.3` on aarch64
  as well (as Debian's arm64 package), removed again on uninstall;
  no errors from rpmlint.
- The LAPACK test suite (`interfaces/lapack/reference/build.sh ...
  test`) fails on numerical errors except the borderline cases listed in
  `known-failures.txt`, on failed error exits and on unfinished test
  programs; `test-all` also on the listed cases.  It runs on Windows
  (MSYS2, Minimal SYStem 2) against `libulmlapack.dll`.

### Fixes

- `?laswp` (the LAPACK function of ulmBLAS) with `K1 > 1` read the pivots
  from `IPIV(1)` instead of `IPIV(K1)`.
- Complex `?scal`: equal elements of x could get different results,
  depending on their position (the vectorized loop and the scalar loops
  contracted the complex product into FMA (fused multiply-add) in
  different ways; seen with the AVX2 (Advanced Vector Extensions 2)
  backend).  The product is now written with `std::fma` where the
  hardware has FMA.  LAPACK's eigenvalue tests CEV, CVX and CGV failed
  because of it.
- Apple Silicon, SME (Scalable Matrix Extension) backend: complex GEMM
  (general matrix-matrix product) for m, n <= 8 used a NEON kernel that
  sums the products of the complex multiplication in another order than
  the SME kernels, so an element of C depended on m and n (LAPACK's tests
  CEV, CGV, ZEV and ZGV failed).  Now the SME kernels for all sizes.
- MSVC (Microsoft Visual C++): no warning C4324 for the aligned counters
  of the thread pool and of GEMM (padded explicitly).

## 0.4.1 (2026-11-18)

### New functions

- GEMM (general matrix-matrix product) with A and B in bfloat16 or IEEE
  (Institute of Electrical and Electronics Engineers) half precision and C
  in float: `cblas_sbgemm`, `sbgemm_`, `cblas_shgemm`, `shgemm_` (the
  signatures of OpenBLAS; PyTorch calls them if the BLAS (Basic Linear
  Algebra Subprograms) it is built with has them), and
  `ulmblas_sbgemm_strided`, `ulmblas_shgemm_strided` with arbitrary
  strides.
- Integer GEMM, exact, C in int32: `ulmblas_i8i32gemm_strided` (int8, A
  and B signed or unsigned), `cblas_gemm_s8u8s32` and
  `cblas_gemm_s16s16s32` (signatures and semantics of oneMKL, the oneAPI
  Math Kernel Library), `ulmblas_i16i32gemm_strided`.  Kernels: SME2
  (Scalable Matrix Extension 2) on Apple M4/M5, NEON SDOT/UDOT (signed
  and unsigned dot product) and I8MM (8-bit integer matrix multiply),
  AVX-VNNI and AVX512-VNNI (Advanced
  Vector Extensions, Vector Neural Network Instructions), AVX2.
- `interfaces/python/ulmblas.py`: these products for NumPy arrays of any
  strides, without copies.
- Thread control: `ulmblas_set_affinity`, `ulmblas_get_affinity`
  (Fortran with suffix `_`), environment `ULMBLAS_AFFINITY` and
  `ULMBLAS_SPIN_US`; `ulmblas_set_num_threads` and `ULMBLAS_NUM_THREADS`
  (otherwise `OMP_NUM_THREADS`) now also apply to the level 3 functions
  (see INSTALL.md, Threads).

### Performance

- Level 3: GEMM, GEMMT, SYRK/HERK, SYR2K/HER2K, SYMM/HEMM, TRSM and TRMM
  run on one blocked algorithm with selection tables per CPU (central
  processing unit): kernel and block sizes chosen by problem size,
  generated by `make tune`; tables for Apple M5, Intel Ice Lake, Intel
  Alder Lake and AMD (Advanced Micro Devices) Zen 4, all four types.
- Level 3 multithreading on an own thread pool (POSIX (Portable
  Operating System Interface) threads, Windows threads; no OpenMP (Open
  Multi-Processing) needed), with optional
  pinning of the threads.
- All vendored micro kernels of BLIS (BLAS-like Library Instantiation
  Software), AOCL-BLAS (AMD Optimizing CPU Libraries) and OpenBLAS can
  be selected for float, double, complex float and complex double; the
  x86-64 backends use the fast SGEMM, CGEMM and
  ZGEMM kernels.
- Kernels for small matrices: SME2, NEON, AVX2, AVX-512 and the OpenBLAS
  SKYLAKEX kernels, selected by problem size.
- SME2: the macro kernel runs in one streaming section per block (double
  and float).
- getrf (LU (lower-upper) factorization with partial pivoting): lookahead
  on the thread pool (the next panel is factorized while the other
  threads update the trailing matrix); `getrfSelect` chooses recursive,
  lookahead or tasks (`OPENMP=1`) by size and threads.
- Thresholds, block sizes and thread counts (threads of GEMM, the
  variants and diagonal blocks of TRSM and TRMM, threads of GEMV and GER,
  the tiles of potrf) in one place, `ulmblas/config/tuned.h`, measured
  per platform by `make tune-params`; values for Apple M5, Intel Ice
  Lake, Intel Alder Lake and AMD Zen 4.
- Level 2: fused GEMV and GER kernels, threads for GEMV and GER, blocked
  TRSV.
- Level 1: vectorized loops for axpy, copy, scal, swap, rot, rotm, asum,
  dot, sdsdot, nrm2 and iamax.

### Memory

- All internal buffers are bounded by constants, independent of the
  problem size: packing buffers, LAPACK (Linear Algebra PACKage) copies,
  pivots, task tokens, the kernels for small matrices and integer GEMM
  (blocks of 1024 x 1024).  Out of memory still terminates the program
  with a message.

### API (application programming interface) changes

- C++ template interface: `getrs` and `getri` take a
  trailing argument `pivBase` (default 0; 1 for the pivots of LAPACK).
  Existing calls compile unchanged.  The C and Fortran 77 interfaces are
  unchanged.
- Windows: the Fortran 77 names are also exported as `DGEMM` and `dgemm`.

### Libraries and packaging

- Release tarballs contain only what builds, installs and tests the
  library (about 1 MB (megabyte) compressed instead of 17 MB): no CUDA
  (Compute Unified Device Architecture) tutorial, benchmark results,
  experiments, benchmark programs (`bench/`), tuning (`tune/`), reference
  BLAS for benchmarks (`refblas/`), CI (continuous integration) or
  drafts; `VERSION.txt` is filled in by `git archive`; reproducible with
  `tools/release/make-tarball.sh`.
- `interfaces/install-shared.sh`: `DESTDIR`, `ULMBLAS_LIBDIR`,
  `ULMBLAS_BINDIR`, `ULMBLAS_INCLUDEDIR`, `ULMBLAS_RPATH`,
  `ULMBLAS_STATIC_LIBSTDCXX`, `ULMBLAS_BACKENDDIR`, `ULMBLAS_STEP=build`
  for distribution packages; the version without Git.
- ILP64 (64-bit integer) interfaces in the shared libraries: the names of
  the reference BLAS and CBLAS (C interface to the BLAS) built with `BUILD_INDEX64_EXT_API`
  (`dgemm_64_`, `cblas_dgemm_64`, ...), which the `liblapack.so.3` of Arch
  and openSUSE calls; `ILP64=0` omits them.
- Linux x86-64: `ULMBLAS_SONAMES=yes` also builds `libblas.so.3` and
  `libcblas.so.3` for the alternatives of the distributions.
- x86-64: new backend `generic` (portable kernels) for CPUs without SSE3
  (Streaming SIMD Extensions 3; SIMD: single instruction, multiple data);
  `avx512` requires AVX-512DQ and AVX-512VL.
- Apple Silicon: the SME backend only with a streaming vector length of
  512 bits.
- The dispatching library finds its backends via `ULMBLAS_BACKEND_DIR`,
  the directory of the build, the resolved path or the loaded path.
- Recipes for Debian, Fedora, openSUSE, Arch, Alpine, nixpkgs, Spack,
  EasyBuild, Homebrew and MSYS2 (Minimal SYStem 2) in `packaging/`.
- Windows: `libulmlapack.dll` (Reference-LAPACK on top of ulmBLAS, with
  `ulmlapack.lib` for MSVC, Microsoft Visual C++) built by
  `install-shared.sh` with a Fortran compiler.
- `CITATION.cff` with the current release.

### Fixes

- The shared library is linked with `-pthread` (glibc (GNU C Library;
  GNU: GNU's Not Unix) before 2.34).
- The assembly kernels of OpenBLAS are no longer exported from the
  libraries.
- NEON kernels for float and complex: `std::ptrdiff_t` instead of `long`
  (Windows on ARM64, the 64-bit Arm architecture).
- GCC (GNU Compiler Collection): no redefinition of the inline macro of
  level 1.
- Debian/Ubuntu packages are built without `-Bsymbolic-functions`, so
  that programs can replace `xerbla_`.
- `make clean` also works where there is no dispatching library (Linux on
  aarch64).
- Memory pool: 32-byte alignment with AVX, 64-byte blocks with AVX-512
  (aligned loads of the kernels).
- TRSM: no division by padding zeros of the diagonal tiles, unit diagonal
  without division.

## 0.4.0 (2026-11-11)

Shared libraries with the standard BLAS interfaces (`make
install-shared`), backends selected at load time, Windows DLL (dynamic-link
library), `?gemmtr`; see
[release-notes-0.4.0.md](https://github.com/michael-lehn/ulmBLAS/blob/master/packaging/release-notes-0.4.0.md).
