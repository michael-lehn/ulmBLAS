# OpenBLAS DGEMM on Intel Ice Lake: what runs, and how it differs from BLIS

Why is OpenBLAS 0.3.34 faster than BLIS 2.0 on one core of an Intel Xeon
Platinum 8358 (Ice Lake) — 93-97 against 77-79 GFLOP/s for n ≥ 1024 in
[010-2026-10-26-uc3-icelake-dgemm](../../../benchmark_results/010-2026-10-26-uc3-icelake-dgemm/README.md)
— and can ulmBLAS reach it with OpenBLAS's micro kernel?  This page records
the analysis; the kernel integration is described in
[ulmblas/level3/ukernel/openblas](../../../ulmblas/level3/ukernel/openblas/README.md).
The integrated kernel passes the ulmBLAS test suite on Ice Lake and Zen 4.

## What actually runs

Not inferred from file names, but checked on the library built for and on
an Ice Lake node (`execution-trace.txt`, job scripts `obtrace.sbatch`,
`obargs.sbatch`):

- `openblas_get_corename()`: `SKYLAKEX`; configuration `OpenBLAS 0.3.34
  NO_LAPACK NO_LAPACKE NO_AFFINITY SKYLAKEX SINGLE_THREADED`.
- `KERNEL.SKYLAKEX` builds `dgemm_kernel.o` from
  `kernel/x86_64/dgemm_kernel_16x2_skylakex.c` (the `.S` variant is only
  for clang on Windows).  `BROADCAST_KERNEL` is not defined anywhere in
  OpenBLAS 0.3.34, so the "duplicate and pair" variant of the kernel is
  compiled.  The object in `libopenblas.a` contains 383 `vfmadd231pd`, 87
  `vmovddup`, 66 `vbroadcastf32x4`, 105 `vunpcklpd` and 105 `vunpckhpd`,
  the same instruction mix as the source compiled outside OpenBLAS.
- Under gdb, a 2048 x 2048 x 2048 `cblas_dgemm` calls
  `cblas_dgemm → dgemm_nn → dgemm_kernel` (the level 3 driver
  `driver/level3/level3.c`, not the small-matrix kernels
  `dgemm_small_kernel_*`, which OpenBLAS also builds for SKYLAKEX).

## The kernel

`dgemm_kernel(m, n, k, alpha, A, B, C, ldc)` computes C ← C + alpha·A·B for
a packed block (beta is applied before by `dgemm_beta`).  It is a macro
kernel with 16 x 12 tiles (remainders 8, 4, 2, 1 rows; 10, 8, 6, 4, 2, 1
columns):

| | OpenBLAS SKYLAKEX 16 x 12 | BLIS skx 16 x 14 |
|---|---|---|
| Accumulators | 24 zmm (zmm8-31) | 28 zmm (zmm4-31) |
| A per k | 4 `vmovddup` (16 rows as duplicated pairs) | 2 `vmovapd` |
| B per k | 6 `vbroadcastf32x4` (a pair of columns per 128-bit lane) | 14 `vbroadcastsd` |
| FMAs per k | 24 | 28 |
| Loads per FMA | 0.42 | 0.57 |
| Store | de-interleave with `vunpcklpd`/`vunpckhpd`, `vfmadd213pd` with C | `vmulpd` alpha, `vfmadd231pd` beta·C |
| Packing of B | panels of 2 columns (`gemm_ncopy_2`) | panels of 14 columns |
| Prefetch | A 384/448 bytes ahead; C into L2 during the k loop, into L1 in the last 16 iterations; next block of B into L2 | A, B 4 iterations ahead into L1; C during the last 14+2 iterations; next B panel into L2 |

## The driver

`driver/level3/level3.c`, single threaded, observed for n = 2048 and 4000
(`obargs.sbatch`: arguments of every kernel call):

- **Block sizes** P (MC) = 192, Q (KC) = 384, R (NC) = 8640 (`param.h`,
  SKYLAKEX, x86-64); for n ≤ 8640 the whole width of B is one block.
- **Balanced remainders**: if the rest of k is between Q and 2Q, it is
  split into two halves (k = 2048: 5 x 384 + 2 x 256; k = 4000: 9 x 384 +
  2 x 272), likewise in m (2048 rows: 10 x 192 + 2 x 160).  BLIS and
  ulmBLAS use fixed block sizes and one smaller last block.
- **B packed in chunks interleaved with the first kernel calls**: for the
  first block of A, OpenBLAS packs 12 columns of B and immediately calls
  the kernel for them (about three quarters of all calls are
  `m = 192, n = 12`); only for the following blocks of A it calls the
  kernel with the whole packed width (`n = 2048`).  The packed B is thus
  used while it is still in cache.  BLIS and ulmBLAS pack the whole block
  of B first.
- **beta** is applied to C once, before the loops.

## Answers

Measured with the integrated kernel on the same CPU:

- **OpenBLAS against BLIS**: the kernel.  Inside ulmBLAS, OpenBLAS's kernel
  is 1.13 times as fast as BLIS skx 16 x 14
  ([014-2026-10-27-uc3-icelake-openblas-kernel](../../../benchmark_results/014-2026-10-27-uc3-icelake-openblas-kernel/README.md));
  alone on packed panels it reaches 100 GFLOP/s, BLIS skx 87.5
  ([016-2026-10-27-uc3-icelake-openblas-kernel-prof](../../../benchmark_results/016-2026-10-27-uc3-icelake-openblas-kernel-prof/README.md)).
- **ulmBLAS with OpenBLAS's kernel against native OpenBLAS** (0.92-0.95
  with one call per tile): not the kernel, but the driver.  Calling the
  kernel once per 16 x 12 tile instead of once per block costs 2-3%; the
  separate packing phase (2-6% of the time in ulmBLAS) costs the rest,
  while OpenBLAS overlaps packing B with computation.  Without packing,
  the kernel called per block is as fast as native OpenBLAS.
- **Tuning iteration 2** confirms the first part: with one call per panel
  of B, ulmBLAS reaches 0.962 (0.956 … 0.970) of native OpenBLAS for
  n ≥ 1000, 1.023 times the per-tile calls
  ([020-2026-10-27-uc3-icelake-openblas-kernel-panel](../../../benchmark_results/020-2026-10-27-uc3-icelake-openblas-kernel-panel/README.md)).
- **Tuning iteration 3**, packing B interleaved with the first block of A
  as OpenBLAS does, gains only 0.8%: 0.970 (0.965 … 0.975) of native
  OpenBLAS, 0.992 of oneMKL
  ([021-2026-10-27-uc3-icelake-openblas-kernel-interleave](../../../benchmark_results/021-2026-10-27-uc3-icelake-openblas-kernel-interleave/README.md)).
  Most of the remaining 3% is not the order of packing and computing;
  not measured are the packing code itself (OpenBLAS's SKYLAKEX copy
  kernels against ulmBLAS's generic C++) and OpenBLAS's balanced
  remainders.

**Can ulmBLAS reach OpenBLAS's performance with OpenBLAS's micro kernel?**
Within 3% on Ice Lake (target: 90%, preferably within 5%).  The kernel
alone is as fast inside ulmBLAS as in OpenBLAS; what separates them is in
the driver: the granularity of the kernel calls (2.3%), the order of
packing (0.8%) and, presumably, the packing code.
- Block sizes are not the reason: ulmBLAS uses OpenBLAS's P, Q, R.
