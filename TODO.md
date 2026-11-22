# TODO

## Possible improvements for the next release

- AVX2 backend built with GCC 16.2: borderline LAPACK eigenvalue ratios
  that GCC 12 (heim) and the generic and sse3 backends don't show (CST,
  DST, SST, STFSM in interfaces/lapack/reference/known-failures.txt).
  The avx2 backend built with `-ffp-contract=off` (no FMA (fused
  multiply-add) contraction by the compiler) passes CST, DST and STFSM
  (Linux x86-64, GCC 16.2.1; SST on Windows not checked).  OpenBLAS and
  BLIS keep contraction on (BLIS: `-ffp-contract=fast` for its reference
  kernels), so this is no reason to switch it off.  Open: which functions
  GCC 16 contracts differently from GCC 12.
- Julia: a complete LAPACK in the library for libblastrampoline
  (interfaces/blas/lbt), as OpenBLAS does: Reference-LAPACK compiled with
  64-bit integers and the suffix `_64_` (CMake `BUILD_INDEX64_EXT_API`),
  the own getrf, potrf, ... replacing the reference versions (as in
  `libulmlapack` of `make install-shared`), plus the CBLAS (C interface to
  the BLAS) names Julia calls (`cblas_zdotc_sub`, `cblas_cdotc_sub`), so
  that `UlmBLAS.forward()` can replace OpenBLAS completely (`clear=true`).
  Needs a Fortran compiler in the Yggdrasil recipe.  Before: benchmarks of
  qr, eigen and svd against OpenBLAS (Reference-LAPACK on ulmBLAS was
  slower for dgetrf in experiment/lapack, before the own getrf), and
  Julia's LinearAlgebra tests without OpenBLAS.
- `libulmblas64` (64-bit integers under the usual names, 0.4.2): not
  yet packaged for Debian (as an alternative of `libblas64.so.3`) and
  openSUSE (`ULMBLAS_INTERFACE64=no` in their recipes), Nix (`blas64 =
  true`) not yet supported; not built on Windows; the front of the
  dispatching library has no C functions with indices
  (`ulmblas_dgemm`, `ulmblas_*_strided`; the backends keep their ILP64
  versions internal).  Optionally a suffix option in FlexiBLAS (issue
  #27), so that `libflexiblas64` could use `libulmblas` directly.
- LAPACK (Linear Algebra PACKage) functions with neither stride 1: every
  buffer is bounded (at most KC NC elements copied,
  investigations/2026-11-bounded-memory), so potrf, getrf, getri above
  that size work on the strides (10 to 35 % slower than with one copy of
  the whole matrix).  Candidate: an optional interface with work space
  provided by the caller plus a work space query, which allows copies of
  the whole matrix.
- Multithreaded gemm, pinning with fewer threads than cores: compact pinning
  (one thread per core first, `threads::CpuOrder`) fills the cores of socket
  0 first, so half the threads run on one socket (one memory interface, one
  L3 (level 3) cache).  For large matrices this is slower than unpinned,
  e.g. Ice Lake (2 x 32 cores) dgemm n = 4000 on 32 threads: pinned 1816,
  unpinned 2045 GFLOPS (giga floating-point operations per second); Zen 4 (2
  x 48 cores) on 48 threads: 2109 vs 2434.  Candidate: spread the threads
  over the sockets (or L3 domains) when there are fewer threads than cores.
- Default number of threads with SMT (simultaneous multithreading): the
  pool starts one thread per logical CPU (as OpenBLAS; MKL (Intel Math
  Kernel Library) one per physical core).  On heim (6 cores, 12 logical
  CPUs) 12 threads are clearly slower than 6 for gemm, e.g. dgemm 96 x 96
  x 96: 64 against 83 GFLOPS.  potrf with the default threads (Ice Lake
  128, Zen 4 192; ulmBLAS's potrf and Reference-LAPACK's on ulmBLAS) is
  erratic for n >= 1024: spotrf U on Ice Lake 120 GFLOPS at n = 4096
  against 285 at n = 2048, Reference-LAPACK's dpotrf L on Zen 4 28 GFLOPS
  at n = 4096.  With one thread per core (`ULMBLAS_NUM_THREADS` 64 and 96)
  the values are smooth: spotrf U on Ice Lake 842 GFLOPS at n = 4096,
  spotrf L on Zen 4 266 instead of 29 at n = 1024, dgetrf on Zen 4 477
  instead of 353 at n = 4096.  Candidate: one thread per core by default
  (`ULMBLAS_NUM_THREADS` still overrides), measured on heim, Ice Lake and
  Zen 4.
- `make tune-params` (tune/README.md, ulmblas/config/tuned.h): the
  decisions are fitted to a fixed set of problems per group; a decision by
  m n k for the threads of gemm is also used for trsm, trmm, syrk, symm
  (work m^2 n / 2 etc.) without measurements of these.  Candidates: measure
  the level 3 operations themselves; the partition of the loops of gemm by
  size (only the global constants of BLIS (BLAS-like Library Instantiation
  Software; BLAS: Basic Linear Algebra Subprograms) are measured; the
  defaults stayed on all platforms); the fuse factors, lanes and blocks of
  level 2 (compile-time constants); tiles and threads of getrf
  (ulmblas/lapack/tasks/param.h, getrfParam).
- syrk/herk against MKL (Intel Math Kernel Library) on Ice Lake, 64 threads
  (224cddc4): dsyrk and zherk on par (0.85-1.1), ssyrk 0.68-0.77 (n >=
  4000), cherk about 0.5 (e.g. n = 8000, k = 256: 3768 vs 7630 GFLOPS).
  ssyrk reaches about 65 % of the rate of sgemm, dsyrk 80 % of dgemm:
  candidates the diagonal tiles (local tile and masked update) and the
  balance of the triangle over the threads.
- trsm with SME2 (Scalable Matrix Extension 2, Apple M5): the diagonal tiles
  call the micro kernel one tile at a time (one streaming section each) and
  solve 32 x 32 tiles in scalar code: dtrsm m = 256, n = 2000: 53 GFLOPS
  against 443 of dgemm. Candidate: invert the diagonal tiles once and
  multiply in streaming mode.
- trsm against MKL on Ice Lake, 64 threads: the shapes of LAPACK (Linear
  Algebra PACKage) 256 x 8000 of getrf 0.67 (1196 against 1779 GFLOPS),
  8000 x 256 of potrf on par, both ahead of OpenBLAS; small n x n behind
  MKL (dtrsm 500 x 500: 526 against 1072, 1000 x 1000: 1099 against 1392).
  One thread: 54 against 77 (getrf shape).  Candidates: a left-looking variant (as
  OpenBLAS's trsm_R.c) for A on the right and row-major B (strsm 4000 x
  4000 on the right, one thread: 77 against 149 of OpenBLAS).
- trsm and trmm, A on the left, many threads (the variants of
  `ulmblas/level3/gemm/trshared.h`): the default (shared packing above 4
  diagonal blocks, 12 on Zen 4) is now measured per platform (make
  tune-params).
  Gaps to OpenBLAS: Zen 4 (96 threads) dtrmm 1000 x 1000 0.82 (1532
  against 1871), dtrsm 0.92 (1584 against 1725), 500 x 500 0.77-0.85;
  Ice Lake dtrmm 500 x 500 0.75 (692 against 927).  The columns variant splits B in panels of NR (16 x 12 kernel: 84
  panels for n = 1000, 125 for 1500 on 96 threads: 1 or 2 per thread);
  the diagonal blocks of the shared variant (one panel per thread, short
  products and the substitution) are slow on Zen 4 (strsm 2000 x 2000:
  3128 against 4818 of the columns variant).  Candidates: the frame chosen
  for the columns of one thread (m x n/nt x m) instead of m x n x m; the
  diagonal block in a lookahead (overlapped with the update).
- potrf: the recursive potrf is ahead of OpenBLAS on Ice Lake (64 threads,
  1.2-1.7 for double) and Zen 4 (96 threads, 2-5), about on par on heim (6
  threads).  Open (branch potrf-study, investigations/2026-11-potrf):
  - lookahead on the thread pool for small teams: heim 1.03-1.23 of the
    recursive potrf, slower on 32 and more threads; thread count capped by
    n for small n (Ice Lake n = 1000: 1.6-1.8).  The selection (PotrfPlan)
    by threads, type, layout and n from make tune.
  - MKL on Ice Lake, double, n = 2000-4000: 2.2-2.7 ahead (dpotrf L 4000:
    876 against 2172 GFLOPS); n = 16000 0.85-0.91.  First measure where the
    time goes (critical path through the diagonal blocks or the rate of
    the update).
  - Zen 4 complex n = 2000-4000: 0.5-0.98 of MKL.
- getrf: lookahead on the thread pool (getrfla.tcc, getrfSelect) ahead of
  OpenBLAS on Ice Lake (64 threads) and Zen 4 (96 threads); heim (6 threads)
  0.7-0.93 for n <= 8000 (d 1000: 135 against 172 GFLOPS, z 1000: 189
  against 265), investigations/2026-11-getrf.  Candidates: the trailing
  update on the gemm frame with a sub-team (each chunk packs its rows of L
  again, about 10 % at d 8000); a faster serial panel (critical path for n
  <= 2000 and tall matrices, z panel 0.8 of OpenBLAS); a better kernel for
  one thread in a region (53 against about 60 GFLOPS); per type limits and
  tasks against lookahead (OPENMP=1) by make tune.
- Size of the shared libraries: internal helpers with internal linkage
  (static templates such as packPanelA, .isra clones) are copied into
  every object file by GCC; with the ILP64 interfaces (ILP64=1, default)
  a backend grows from 9.7 to 13.3 MB on Linux (the new functions are
  0.7 MB of it).  Candidate: inline instead of static, which also shrinks
  the LP64 library.
- Level 1, heim (Intel i5-12500, one core), elements per ns (nanosecond)
  against OpenBLAS 0.3.34 (one thread); the other routines are on par or
  ahead for n = 1000, 100000 and 10^7:
  - complex dot (cdotu, cdotc, zdotu, zdotc) for vectors in the caches:
    zdotc n = 1000 3.8 against 6.2.  GCC 12 does not keep the partial sums
    in registers or deinterleaves x and y; the kernel with one permutation
    per vector (ulmblas/level1/kernel/reduce.h) reaches 4.5 without the
    peeled first elements.  Candidates: other loop shapes, or a kernel with
    intrinsics (to be decided).
  - sdsdot, dsdot (products and sum in double): n = 1000 7.6 against
    10.1; the conversions from float limit the loop.
  - sasum n = 100000 (in the L2 (level 2) cache): 22.4 against 38.8
    (dasum is on par).
  - sscal, dscal n = 1000: 36 and 17 against 44 and 25 (in blocks of 32
    with peeled first elements; one loop was 13 for dscal), caxpy n = 1000
    7.1 against 8.0, srot, drot n = 1000 about 0.9 of OpenBLAS.
  - short vectors (n = 100): fixed costs per call, isamax 5.1 against 11.0,
    sdot 8.9 against 12.1, complex dot 1.5 to 1.9 against 4.2 to 5.5.  The
    copy loops for n = 100 depend on the placement of the code: dcopy 9.0
    or 5.7 (OpenBLAS 11.9) with identical machine code at two addresses
    16 bytes apart.
  - multithreading: OpenBLAS with 6 threads at n = 10^7: ddot 2.1, dasum
    4.8, dscal 2.2, daxpy 1.3 elements per ns, ulmBLAS (one thread) 1.4,
    3.2, 1.7, 1.1 (dcopy is not threaded in OpenBLAS).
  - strided vectors: the reductions use partial sums now (2 to 4 times
    faster than before), but sdot, ddot with increment 2 are still slower
    than OpenBLAS (n = 1000: 2.1 and 2.6 against 3.3 and 3.8).  A few
    strided loops are 5 to 15 % slower than before with unchanged source
    (daxpy, dcopy n = 1000; srotm, drotm, sdsdot n = 100), for dcopy with
    identical machine code (placement of the loop).
  - the loop shapes depend on the compiler: GCC 12 needs explicit partial
    sums, blocks and peeled first elements, clang 21 (Apple M5) is faster
    with the simple loops of before for float sums, scal, swap, rot and
    copy (ULMBLAS_LEVEL1_SIMPLE_LOOPS in ulmblas/level1/kernel/simd.h).
- Level 2 (gemv, ger, trsv), heim, against OpenBLAS 0.3.34 on 1 and 6
  threads (investigations/2026-11-level2): the real types on par from n =
  500, threads included.  Remaining:
  - complex gemv for n <= 1000: zgemv with A conjugate transposed 0.51-0.82
    (6 threads 0.53-0.59), zgemv_t n = 100 0.67 (6 threads 0.50-0.83),
    zgemv_n 0.76-0.91.  The loops move by up to 30 % with changes elsewhere
    in the library (placement of the code).  Candidates: the loop shapes of
    OpenBLAS's zgemv kernels, one copy of x for all threads, a profiler.
  - ztrsv with A not transposed: 0.45-0.54 at n = 100, 0.73-0.8 for n =
    500-1000; the block size and fused columns in the diagonal blocks did
    not change it.
  - trsv on threads (OpenBLAS: none): the gemv of the column panels would
    pay for n >= 1000 in the L3 cache.
  - n = 100: sgemv_n 0.79, dgemv_n 0.85, dtrsv_ln 0.72.
  - make tune for the fuse factors, lanes and blocks of
    ulmblas/config/level2.h (the thresholds of the threads are measured by
    make tune-params).
  - Apple M5 (clang, against OpenBLAS 0.3.34 of Homebrew, 1 and 4 threads):
    ahead or on par except A out of the caches (n = 4000): dgemv_n 0.61
    (4 threads 0.62), dgemv_t 0.75 (one thread slower than before: 12.4
    against 15.1), zgemv_n 4 threads 0.49; sgemv_n 0.79-0.82 for n <= 1000;
    ztrsv_lt/ut n = 100 0.67-0.73; dtrsv_ut n = 100 slower than before
    (12.8 against 21.1, still 1.3 of OpenBLAS).
    Other compilers (and GCC on AArch64, clang on x86-64) are not measured.
    On the M5 (NEON) a few cases are about 10 % slower than before with
    unchanged loops: dcopy with increment 2, n = 100.
