ulmBLAS 0.4.1

Level 3 BLAS (Basic Linear Algebra Subprograms) and the LAPACK (Linear
Algebra PACKage) factorizations on one blocked algorithm with an own
thread pool, tuned per CPU (central processing unit).

- Level 3: GEMM (general matrix-matrix product), GEMMT, SYRK/HERK,
  SYR2K/HER2K, SYMM/HEMM, TRSM and TRMM on one blocked algorithm;
  kernels and block sizes selected by problem size from tables measured
  by `make tune` for Apple M5, Intel Ice Lake, Intel Alder Lake and AMD
  (Advanced Micro Devices) Zen 4, all four types.
- Threads: an own thread pool (POSIX (Portable Operating System
  Interface) threads, Windows threads), optional pinning
  (`ULMBLAS_AFFINITY`); thresholds and thread counts measured per
  platform (`make tune-params`).
- getrf (LU (lower-upper) factorization with partial pivoting) with
  lookahead on the thread pool; potrf, getrs, getri on bounded buffers.
  dgetrf, n = 4000: Intel Ice Lake (64 threads) 519 GFLOPS (giga
  floating-point operations per second) against 211 of OpenBLAS, AMD
  Zen 4 (96 threads) 827 against 162.
- Julia (`UlmBLAS.forward()`, Apple M5, 10 threads, n = 4000): `lu` 357
  GFLOPS against 177 with Julia's default OpenBLAS, `cholesky` 374
  against 189.
- Level 2 and level 1: fused GEMV (matrix-vector product) and GER kernels
  with threads, vectorized level 1 loops.
- New: GEMM in bfloat16 and IEEE (Institute of Electrical and Electronics
  Engineers) half precision (`cblas_sbgemm`, `cblas_shgemm`), integer
  GEMM (`cblas_gemm_s8u8s32`, `cblas_gemm_s16s16s32`,
  `ulmblas_i8i32gemm_strided`), kernels for small matrices.
- ILP64 (64-bit integer) names in the shared libraries (`dgemm_64_`,
  `cblas_dgemm_64`), as used by the `liblapack.so.3` of Arch and openSUSE.
- All internal buffers are bounded by constants, independent of the
  problem size.
- Release tarball of about 1 MB (megabyte); recipes for Debian, Fedora,
  openSUSE, Arch, Alpine, nixpkgs, Spack, EasyBuild, Homebrew and MSYS2
  (Minimal SYStem 2) in `packaging/`.

All changes: CHANGELOG.md.  Packages: packaging/README.md.
