ulmBLAS 0.4.2

Error handlers set at run time, ulmBLAS as a FlexiBLAS backend, the LAPACK
(Linear Algebra PACKage) functions of ulmBLAS in the shared library, and a
library with 64-bit integers under the usual names.

- Error handlers: `cblas_set_xerbla`, `cblas_get_xerbla`,
  `set_blas_xerbla_` and `get_blas_xerbla_` with the interface of
  Reference-LAPACK pull request #1407, also under the ILP64 (64-bit
  integer) names.  A program that defines `xerbla_` or `cblas_xerbla`
  still replaces them.
- FlexiBLAS: `flexiblas_set_num_threads` sets the threads of ulmBLAS, and
  FlexiBLAS passes the program's `XERBLA` to it.  Two fixes on the
  FlexiBLAS side are proposed: mpimd-csc/flexiblas#122, without which
  `flexiblas_get_num_threads` does not find ulmBLAS's getter, and #123
  for the `XERBLA` of the program on FlexiBLAS's master (3.5.0 finds it).
  The Fedora package `flexiblas-ulmblas` registers ulmBLAS as a backend.
- `libulmblas` exports the LAPACK functions of ulmBLAS: `?getrf`,
  `?getrs`, `?gesv`, `?getri`, `?potrf`, `?potrs`, `?posv`, `?potri`,
  `?trtrs`, `?trtri`, `?lauum` and `?laswp` for all four types.
  `libulmlapack` (Reference-LAPACK on top of ulmBLAS) and FlexiBLAS use
  them instead of the reference versions.
- `libulmblas64`: the interfaces with 64-bit integers under the usual
  names (`dgemm_`, `cblas_dgemm`, ...), as Debian's `libblas64`, for
  programs compiled with 64-bit `INTEGER` and as the backend of
  `libflexiblas64`.
- `libulmlapack` is built with the makefiles of LAPACK: `make
  install-shared` no longer needs CMake.
- The libraries link no C++ runtime (no `libstdc++`, no `libc++`): C++ is
  only the language of the implementation, the interfaces are C and
  Fortran.
- The build is one make: no make calls another, `make -j` builds
  everything with one job server.  New targets `build-shared`,
  `install-shared` and `check-package` for the build, installation and
  test step of a package.
- Fixes: `?laswp` with `K1 > 1`; complex `?scal` gave equal elements
  different results depending on their position (AVX2 (Advanced Vector
  Extensions 2) backend); complex GEMM (general matrix-matrix product)
  with the SME (Scalable Matrix Extension) backend for m, n <= 8
  depended on m and n.
- Tests: the LAPACK test suite passes except for a list of documented
  borderline cases, also on Windows with MSYS2 (Minimal SYStem 2).
  The ATLAS (Automatically Tuned Linear Algebra Software) testers
  (`make check-large`, 876 runs with one thread and with all threads)
  pass without failures on Apple M5, Intel Alder Lake, Intel Ice Lake and
  AMD (Advanced Micro Devices) Zen 4, with the SME, AVX2, AVX-512
  (Advanced Vector Extensions, 512 bits) and SSE3 (Streaming SIMD
  Extensions 3, SIMD: single instruction, multiple data) backends.
- Guides: four guides on the concepts behind ulmBLAS, each a PDF with
  complete examples, written for the course High Performance Computing I
  at Ulm University: GEMM and the level-3 BLAS (Basic Linear Algebra
  Subprograms), the LU (lower-upper) and Cholesky factorizations,
  vectorization and multithreading with OpenMP (Open Multi-Processing)
  and C++ atomics, and the C++ that ulmBLAS uses.  They are in the
  repository, not in the release tarball:
  https://github.com/michael-lehn/ulmBLAS/tree/v0.4.2/doc
- Packages: Fedora (Copr `mlehn/ulmblas`), openSUSE (`home:lehn`,
  submitted to `science`), Homebrew tap `michael-lehn/ulmblas`; Debian
  and Spack in review.

All changes: CHANGELOG.md.  Packages: packaging/README.md.
