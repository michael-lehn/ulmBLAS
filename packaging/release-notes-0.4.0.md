ulmBLAS 0.4.0

Shared libraries with the standard BLAS interfaces, for NumPy, SciPy and
every other program that links a system BLAS.

- `make install-shared PREFIX=...`: `libulmblas` (Fortran 77 interface
  with 32-bit integers and CBLAS), `libulmlapack` (Reference-LAPACK 3.12.1,
  unmodified, on top of it), `cblas.h`, `ulmblas.pc`, `ulmlapack.pc`.
- Kernels selected at load time: `libulmblas` is a small front that loads
  the backend for the CPU: SME2 or NEON on Apple Silicon, AVX-512,
  AVX2/FMA or SSE3 on x86-64 (Linux, Windows).  `ULMBLAS_BACKEND`
  overrides the choice, `ulmblas_backend()` reports it.
- Windows: `libulmblas.dll` (MinGW-w64), import libraries for MinGW and
  MSVC; depends only on system DLLs.  A program replaces `XERBLA` or
  `cblas_xerbla` by exporting it.
- Without `libstdc++.a` (e.g. RHEL) the C++ runtime is linked
  dynamically.
- `interfaces/python/numpy.sh PREFIX VENV`: NumPy built from source on
  these libraries.  NumPy and SciPy that link `libblas.so.3` (conda-forge,
  Linux distributions) run on ulmBLAS without being rebuilt.
- `?gemmtr` (BLAS of LAPACK 3.12).
- SME: no floating-point exception flags left behind by streaming mode
  (#0031).
- Sources formatted with clang-format.

Packages (conda-forge, Homebrew): see packaging/README.md.
