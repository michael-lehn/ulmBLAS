# GEMM micro kernels

This directory contains the GEMM micro kernels of ulmBLAS: the kernel that
updates an MR x NR tile of C in registers, called by the macro kernels of
all level 3 routines.  The top-level [support matrix](../../../README.md#current-micro-kernel-support)
links to the sections below, one per architecture and data type.

| Directory | Contents | Origin |
|---|---|---|
| [`ref`](ref) | portable reference kernels for all types (also TRSM kernels) | ulmBLAS |
| [`sse`](sse) | SSE3 kernels for double (GEMM, TRSM) | ulmBLAS |
| [`avx`](avx) | AVX2/FMA DGEMM kernel | ulmBLAS port of a BLIS kernel |
| [`neon`](neon) | NEON DGEMM kernel; SGEMM, CGEMM and ZGEMM kernels | ulmBLAS port of a BLIS kernel; SGEMM, CGEMM, ZGEMM: ulmBLAS |
| [`sme`](sme) | SME2 DGEMM kernel and streaming-mode packing; SGEMM, CGEMM and ZGEMM kernels and packing | ulmBLAS, DGEMM kernel derived from BLIS PR #921; SGEMM, CGEMM, ZGEMM: ulmBLAS |
| [`blis`](blis/README.md) | vendored BLIS 2.0 and AOCL-BLAS 5.x kernels, the adapter for all vendored kernels | BLIS, AOCL-BLAS |
| [`openblas`](openblas/README.md) | vendored OpenBLAS 0.3.34 kernels | OpenBLAS |

**Selection.**  Without further flags ulmBLAS selects its own kernel from
the compiler's target macros (`ulmblas/config/simd.h`): SSE3 on x86-64,
AVX/FMA with `ARCHFLAGS='-mavx -mfma'`, NEON on AArch64, SME2 with clang
and `ARCHFLAGS=-mcpu=apple-m4`.  A vendored kernel replaces it for one
data type with exactly one `-DULMBLAS_DGEMM_UKERNEL_<origin>_<name>`
([`blis/select.h`](blis/select.h)), `-DULMBLAS_SGEMM_UKERNEL_…`,
`-DULMBLAS_CGEMM_UKERNEL_…` or `-DULMBLAS_ZGEMM_UKERNEL_…`
([`blis/select_s.h`](blis/select_s.h), [`select_c.h`](blis/select_c.h),
[`select_z.h`](blis/select_z.h)); the four selections are independent.
Its default block sizes are those of the upstream configuration that uses
it.  There is no run-time dispatch:
the CPU must support the instruction set the build selects.

**Status.**  "Tested" means that the complete test suite (`make check`:
numeric tests and the Netlib BLAS and CBLAS test programs) passes with the
kernel on native hardware: the default configurations `testcase01` …
`testcase09` for ulmBLAS's own kernels, and for every vendored kernel its
configurations `blis-NAME`, `openblas-NAME` (DGEMM), `s-NAME`, `c-NAME`,
`z-NAME` and `…-small` (small block sizes).  Test records of the vendored
kernels:
[investigations/2026-10-bwunicluster3/kernels/tests](../../../investigations/2026-10-bwunicluster3/kernels/tests/README.md)
(DGEMM) and
[investigations/2026-10-gemm-datatypes](../../../investigations/2026-10-gemm-datatypes/README.md)
(ZGEMM, CGEMM).
Tested CPUs: Intel Core i5-12500 (Alder Lake: SSE3, AVX, AVX2/FMA), Intel
Xeon Platinum 8358 (Ice Lake-SP: also AVX-512), AMD EPYC 9454 (Zen 4: also
AVX-512), Apple M5 (NEON, SME2).

**License.**  ulmBLAS's own code is under the BSD 3-Clause License of
ulmBLAS.  Kernels derived from or vendored from other projects keep the
copyright notices of their sources; [LICENSE](../../../LICENSE) reproduces
them (section numbers below).

## Portable reference kernel (all types)

`ref/ugemm.tcc`: a C++ loop over the tile, for float, double, complex
float and complex double; `ref/utrlsm.tcc`, `ref/utrusm.tcc` for TRSM.  It
is the universal fallback: used for every type that has no optimized
kernel on the target, on every platform.  The level 3 routines still use
cache blocking and packing with it.  MR x NR: 8 x 4 (float), 4 x 4
(double), 4 x 2 (complex float), 2 x 2 (complex double)
(`ulmblas/config/blocksize.h`).

## DGEMM: x86-64 SSE3

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `sse` | ulmBLAS (inline assembly, 2014) | 4 x 4 | `sse/ugemm.h`, `sse/ugemm.tcc` (also TRSM kernels `sse/utrlsm.*`, `sse/utrusm.*`) | default on x86-64 (SSE3 is part of the default `ARCHFLAGS`) | tested (heim; Rosetta 2) | [001](../../../benchmark_results/001-2026-10-24-heim-avx-fma/README.md), [011](../../../benchmark_results/011-2026-10-26-heim-blis-kernels/README.md) |
| `blis_penryn_4x4` | BLIS 2.0 `bli_dgemm_penryn_asm_4x4` | 4 x 4 | `blis/kernel/penryn_4x4.h` | `-msse3 -DULMBLAS_DGEMM_UKERNEL_blis_penryn_4x4` | tested (heim, also ASan/UBSan; Ice Lake; Zen 4) | [011](../../../benchmark_results/011-2026-10-26-heim-blis-kernels/README.md), [012](../../../benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md), [015](../../../benchmark_results/015-2026-10-27-uc3-zen4-kernels-socket/README.md) (preliminary) |

Requirements: x86-64 (LP64) with SSE3.  License: ulmBLAS; BLIS kernel:
LICENSE section 7.  Details of the BLIS kernel: [blis/README.md](blis/README.md).

## DGEMM: x86-64 AVX

Kernels for AVX without FMA (Sandy Bridge class).

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `blis_sandybridge_8x4` | BLIS 2.0 `bli_dgemm_sandybridge_asm_8x4` | 8 x 4 | `blis/kernel/sandybridge_8x4.h` | `-mavx -DULMBLAS_DGEMM_UKERNEL_blis_sandybridge_8x4` | tested (heim, also ASan/UBSan; Ice Lake; Zen 4) | [011](../../../benchmark_results/011-2026-10-26-heim-blis-kernels/README.md), [012](../../../benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md), [015](../../../benchmark_results/015-2026-10-27-uc3-zen4-kernels-socket/README.md) (preliminary) |
| `blis_sandybridge_int_8x4` | BLIS 2.0 `bli_dgemm_sandybridge_int_8x4` (intrinsics) | 8 x 4 | `blis/kernel/sandybridge_int_8x4.h` | `-mavx -DULMBLAS_DGEMM_UKERNEL_blis_sandybridge_int_8x4` | tested (heim, also ASan/UBSan; Ice Lake; Zen 4) | not benchmarked |

Requirements: AVX.  License: LICENSE section 7.  Details:
[blis/README.md](blis/README.md).

## DGEMM: x86-64 AVX2/FMA

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `avx` | ulmBLAS port of BLIS 2.0 `bli_dgemm_haswell_asm_6x8` (macros expanded, ulmBLAS interface) | 6 x 8 | `avx/ugemm.h`, `avx/ugemm.tcc` | default with `ARCHFLAGS='-mavx -mfma'` | tested (heim: `testcase06`, `testcase07`; Ice Lake; Zen 4) | [001](../../../benchmark_results/001-2026-10-24-heim-avx-fma/README.md), [008](../../../benchmark_results/008-2026-10-26-heim-avx-fma-memory/README.md), [010](../../../benchmark_results/010-2026-10-26-uc3-icelake-dgemm/README.md), [011](../../../benchmark_results/011-2026-10-26-heim-blis-kernels/README.md), [023](../../../benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md) |
| `blis_haswell_6x8`, `blis_haswell_8x6` | BLIS 2.0 `bli_dgemm_haswell_asm_6x8`, `_8x6` | 6 x 8, 8 x 6 | `blis/kernel/haswell_6x8.h`, `haswell_8x6.h` | `-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_6x8` (or `_8x6`) | tested (heim, also ASan/UBSan; Ice Lake; Zen 4) | [011](../../../benchmark_results/011-2026-10-26-heim-blis-kernels/README.md), [013](../../../benchmark_results/013-2026-10-27-heim-blis-kernels-mc/README.md), [012](../../../benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md) |
| `blis_haswell_12x4`, `blis_haswell_4x12` | BLIS 2.0 `kernels/haswell/3/old/` (BLIS 0.x kernels) | 12 x 4, 4 x 12 | `blis/kernel/haswell_12x4.h`, `haswell_4x12.h` | `-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_12x4` (or `_4x12`) | tested (heim, also ASan/UBSan; Ice Lake; Zen 4) | [017](../../../benchmark_results/017-2026-10-27-heim-blis-kernels-old/README.md), [019](../../../benchmark_results/019-2026-10-27-uc3-icelake-kernels-k4/README.md) |
| `openblas_haswell_4x8` | OpenBLAS 0.3.34 `dgemm_kernel_4x8_haswell.S` (assembled unchanged) | 4 x 8 (block calls) | `openblas/asm/haswell_4x8.S`, `openblas/kernel/haswell_4x8.h` | `-mavx -mavx2 -mfma -DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8` | tested (heim, also ASan/UBSan and valgrind) | [024](../../../benchmark_results/024-2026-10-27-heim-openblas-haswell-kernel/README.md), [026](../../../benchmark_results/026-2026-10-27-heim-small-dgemm/README.md) |

Requirements: AVX and FMA (the OpenBLAS kernel: AVX2).  License: LICENSE
sections 2 (`avx`), 7 (BLIS kernels), 8 (OpenBLAS kernel).  Details:
[blis/README.md](blis/README.md),
[openblas/README.md](openblas/README.md#the-haswell-kernel-haswell_4x8).
The BLIS and AOCL-BLAS configurations for AMD Zen 1 to 3 use the haswell
kernels; no such CPU has been tested.

## DGEMM: x86-64 AVX-512

| Kernel | Origin | MR x NR | Files | Selection (with `-mavx -mfma -mavx512f`) | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `blis_skx_16x14`, `blis_skx_16x12_l2` | BLIS 2.0 `bli_dgemm_skx_asm_16x14`, `_16x12_l2` | 16 x 14, 16 x 12 | `blis/kernel/skx_16x14.h`, `skx_16x12_l2.h` | `-DULMBLAS_DGEMM_UKERNEL_blis_skx_16x14` (or `_16x12_l2`) | tested (Ice Lake; Zen 4) | [012](../../../benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md), [014](../../../benchmark_results/014-2026-10-27-uc3-icelake-openblas-kernel/README.md), [023](../../../benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md) |
| `blis_zen4_32x6`, `blis_zen4_8x24` | AOCL-BLAS 5.0 `bli_dgemm_zen4_asm_32x6`, `_8x24` | 32 x 6, 8 x 24 | `blis/kernel/zen4_32x6.h`, `zen4_8x24.h` | `-DULMBLAS_DGEMM_UKERNEL_blis_zen4_32x6` (or `_8x24`) | tested (Ice Lake; Zen 4) | [012](../../../benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md), [015](../../../benchmark_results/015-2026-10-27-uc3-zen4-kernels-socket/README.md) (preliminary), [023](../../../benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md) |
| `blis_zen5_8x24` | AOCL-BLAS 5.1 `bli_dgemm_avx512_asm_8x24` | 8 x 24 | `blis/kernel/zen5_8x24.h` | `-DULMBLAS_DGEMM_UKERNEL_blis_zen5_8x24` | tested (Ice Lake; Zen 4; not run on Zen 5) | [018](../../../benchmark_results/018-2026-10-27-uc3-zen4-kernels-k4-socket/README.md) (preliminary), [019](../../../benchmark_results/019-2026-10-27-uc3-icelake-kernels-k4/README.md) |
| `openblas_skylakex_16x12` | OpenBLAS 0.3.34 `dgemm_kernel_16x2_skylakex.c` (B addressing adapted) | 16 x 12 (panel calls) | `openblas/kernel/skylakex_16x12.h` | `-DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12` | tested (Ice Lake; Zen 4) | [014](../../../benchmark_results/014-2026-10-27-uc3-icelake-openblas-kernel/README.md), [016](../../../benchmark_results/016-2026-10-27-uc3-icelake-openblas-kernel-prof/README.md), [020](../../../benchmark_results/020-2026-10-27-uc3-icelake-openblas-kernel-panel/README.md), [021](../../../benchmark_results/021-2026-10-27-uc3-icelake-openblas-kernel-interleave/README.md), [023](../../../benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md) |

Requirements: AVX-512F.  The Zen 4 benchmarks are preliminary (one socket
of a shared node).  No sanitizer run yet for the AVX-512 kernels.
License: LICENSE sections 7 (BLIS, AOCL-BLAS) and 8 (OpenBLAS).  Details:
[blis/README.md](blis/README.md), [openblas/README.md](openblas/README.md).

## DGEMM: AArch64 NEON

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `neon` | ulmBLAS port of BLIS 2.0 `bli_dgemm_armv8a_asm_8x6r` (operands swapped: C^T = B^T A^T) | 6 x 8 | `neon/ugemm.h`, `neon/ugemm.tcc` | default on AArch64 (LP64); `-DULMBLAS_NO_NEON` selects the reference kernel | tested (Apple M5: `testcase04`, `testcase08`, also ASan/UBSan) | [002](../../../benchmark_results/002-2026-10-24-macbook-m5-neon/README.md), [003](../../../benchmark_results/003-2026-10-25-macbook-m5-profile/README.md) |

On AArch64, DGEMM of small problems (m·n·k ≤ 8192) uses a packing-free
NEON path (`../gemmdirect.tcc`).  Only run on Apple silicon (Apple M5);
Linux on AArch64 is untested.  License: LICENSE section 3; provenance and
modifications in `neon/ugemm.tcc`.

## DGEMM: AArch64 SME2

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `sme` (experimental) | ulmBLAS; the kernel is derived from `bli_dgemm_armsme_int_4SVLx2SVL` of [BLIS PR #921](https://github.com/flame/blis/pull/921), the streaming-mode panel packing is ulmBLAS's | 32 x 16 | `sme/ugemm.h`, `sme/ugemm.tcc`, `sme/pack.h`, `sme/pack.tcc` | clang with SME2 enabled, e.g. `ARCHFLAGS=-mcpu=apple-m4`; `-DULMBLAS_NO_SME` keeps the NEON kernel | tested (Apple M5, Apple clang: `testcase09`, `testcase04`) | [004](../../../benchmark_results/004-2026-10-25-macbook-m5-sme/README.md), [005](../../../benchmark_results/005-2026-10-25-macbook-m5-sme-blocksizes/README.md), [006](../../../benchmark_results/006-2026-10-25-macbook-m5-sme-square/README.md), [007](../../../benchmark_results/007-2026-10-26-macbook-m5-sme-small/README.md), [022](../../../benchmark_results/022-2026-10-27-macbook-m5-sme-blocking/README.md) |

Requirements: SME2 with FEAT_SME_F64F64 and a streaming vector length of
512 bits (Apple M4, M5; tested only on an Apple M5); GCC lacks the Darwin
runtime routine `__arm_get_current_vg`.  Results and open questions:
[PERFORMANCE.md: AArch64: SME2](../../../PERFORMANCE.md#aarch64-sme2-apple-m5-experimental).
License: LICENSE section 4; provenance and modifications in
`sme/ugemm.tcc`.

## ZGEMM: x86-64

Vendored ZGEMM kernels (complex double), selected with exactly one
`-DULMBLAS_ZGEMM_UKERNEL_<origin>_<name>` ([`blis/select_z.h`](blis/select_z.h)),
independently of the DGEMM kernel; without one ZGEMM uses the reference
kernel.  Details (packing, conjugation, row preference, block calls,
vendoring): [blis/README.md](blis/README.md#zgemm-kernels).

| Kernel | Origin | MR x NR | Files | Selection | Status |
|---|---|---|---|---|---|
| `blis_sandybridge_4x4` | BLIS 2.0 `bli_zgemm_sandybridge_asm_4x4` | 4 x 4 | `blis/kernel/z_sandybridge_4x4.h` | `-mavx -DULMBLAS_ZGEMM_UKERNEL_blis_sandybridge_4x4` | tested (heim, also ASan/UBSan and valgrind) |
| `blis_haswell_3x4`, `blis_haswell_4x3` | BLIS 2.0 `bli_zgemm_haswell_asm_3x4`, `_4x3` | 3 x 4, 4 x 3 | `blis/kernel/z_haswell_3x4.h`, `z_haswell_4x3.h` | `-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_haswell_3x4` (or `_4x3`) | tested (heim, also ASan/UBSan and valgrind) |
| `blis_aocl_haswell_3x4` | AOCL-BLAS 5.1 `bli_zgemm_haswell_asm_3x4` (AMD's version, BLIS 0.x interface) | 3 x 4 | `blis/kernel/z_aocl_haswell_3x4.h` | `-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_aocl_haswell_3x4` | tested (heim, also ASan/UBSan and valgrind) |
| `openblas_haswell_4x2` | OpenBLAS 0.3.34 `zgemm_kernel_4x2_haswell.c` | 4 x 2 (block calls, 4 x 6 tiles) | `openblas/kernel/z_haswell_4x2.h` | `-mavx -mavx2 -mfma -DULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2` | tested (heim, also ASan/UBSan and valgrind) |
| `blis_zen4_12x4`, `blis_zen4_4x12` | AOCL-BLAS 5.1 `bli_zgemm_zen4_asm_12x4`, `_4x12` | 12 x 4, 4 x 12 | `blis/kernel/z_zen4_12x4.h`, `z_zen4_4x12.h` | `-mavx -mfma -mavx512f -mavx512dq -DULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4` (or `_4x12`) | tested (Ice Lake, also ASan/UBSan) |
| `openblas_skylakex_4x2` | OpenBLAS 0.3.34 `zgemm_kernel_4x2_skylakex.c` | 4 x 2 (block calls, 4 x 12 tiles) | `openblas/kernel/z_skylakex_4x2.h` | `-mavx -mfma -mavx512f -DULMBLAS_ZGEMM_UKERNEL_openblas_skylakex_4x2` | tested (Ice Lake, also ASan/UBSan) |

Benchmarks, single-threaded, m = n = k = 50 … 4000, n ≥ 1000:

- Intel Core i5-12500
  ([027](../../../benchmark_results/027-2026-10-28-heim-zgemm/README.md)):
  `openblas_haswell_4x2` 0.995 of native OpenBLAS; `blis_haswell_3x4`
  1.003, `blis_haswell_4x3` 0.999, `blis_aocl_haswell_3x4` 1.001 of
  native BLIS (same kernel); `blis_sandybridge_4x4` 0.61 of OpenBLAS.
- Intel Xeon Platinum 8358
  ([028](../../../benchmark_results/028-2026-10-28-uc3-icelake-zgemm/README.md)):
  `blis_zen4_12x4` 0.970 of oneMKL and 0.992 of native AOCL-BLAS (same
  kernel), 1.153 of OpenBLAS; `blis_zen4_4x12` 0.905 of oneMKL;
  `openblas_skylakex_4x2` 0.987 of native OpenBLAS.

Tests: configurations `z-NAME` and `z-NAME-small` in `test/config/`;
test records in
[investigations/2026-10-gemm-datatypes](../../../investigations/2026-10-gemm-datatypes/README.md).
License: LICENSE sections 7 (BLIS, AOCL-BLAS) and 8 (OpenBLAS).

## ZGEMM: AArch64 NEON

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `neon` | ulmBLAS (not derived from another library) | 4 x 3 | `neon/zgemm.tcc`, `neon/ugemm.h` | default on AArch64 (LP64), as for DGEMM | tested (Apple M5: `make check`, also with small block sizes and ASan/UBSan) | [029](../../../benchmark_results/029-2026-10-28-macbook-m5-zgemm/README.md) |

MC, KC, NC = 192, 512, 4080; 24 accumulators (FMLA by element), any
strides of C.  n ≥ 1000: 1.20 of OpenBLAS, 1.11 of Eigen (Apple M5).
License: ulmBLAS.

## ZGEMM: AArch64 SME2

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `sme` (experimental) | ulmBLAS (not derived from another library) | 16 x 16 | `sme/zgemm.tcc`, `sme/zpack.tcc`, `sme/ugemm.h`, `sme/pack.h` | with the SME2 DGEMM kernel (`ARCHFLAGS=-mcpu=apple-m4`) | tested (Apple M5: `make check`, also with small block sizes and ASan/UBSan) | [029](../../../benchmark_results/029-2026-10-28-macbook-m5-zgemm/README.md) |

MC, KC, NC = 384, 1024, 1024.  Real parts in ZA0 … ZA3, imaginary parts
in ZA4 … ZA7, per k 4 LD2D and 16 FMOPA/FMOPS; packing in streaming mode
as for DGEMM.  n ≥ 1000: 1.123 of Apple Accelerate, 1.64 for n = 50 …
250 (Apple M5).  Requirements as for the SME2 DGEMM kernel.  License:
ulmBLAS.

## SGEMM: x86-64

Vendored SGEMM kernels (float), selected with exactly one
`-DULMBLAS_SGEMM_UKERNEL_<origin>_<name>` ([`blis/select_s.h`](blis/select_s.h)),
independently of the DGEMM and ZGEMM kernels; without one SGEMM uses the
reference kernel.

| Kernel | Origin | MR x NR | Files | Selection | Status |
|---|---|---|---|---|---|
| `blis_penryn_8x4` | BLIS 2.0 `bli_sgemm_penryn_asm_8x4` | 8 x 4 | `blis/kernel/s_penryn_8x4.h` | `-msse3 -DULMBLAS_SGEMM_UKERNEL_blis_penryn_8x4` | tested (Ice Lake, also ASan/UBSan) |
| `blis_sandybridge_8x8` | BLIS 2.0 `bli_sgemm_sandybridge_asm_8x8` | 8 x 8 | `blis/kernel/s_sandybridge_8x8.h` | `-mavx -DULMBLAS_SGEMM_UKERNEL_blis_sandybridge_8x8` | tested (Ice Lake, also ASan/UBSan) |
| `blis_haswell_6x16`, `blis_haswell_16x6` | BLIS 2.0 `bli_sgemm_haswell_asm_6x16`, `_16x6` | 6 x 16, 16 x 6 | `blis/kernel/s_haswell_6x16.h`, `s_haswell_16x6.h` | `-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_6x16` (or `_16x6`) | tested (Ice Lake, also ASan/UBSan) |
| `blis_haswell_24x4`, `blis_haswell_4x24` | BLIS 2.0 `bli_sgemm_haswell_asm_24x4`, `_4x24` (kernels/haswell/3/old) | 24 x 4, 4 x 24 | `blis/kernel/s_haswell_24x4.h`, `s_haswell_4x24.h` | `-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_24x4` (or `_4x24`) | tested (Ice Lake, also ASan/UBSan) |
| `blis_aocl_haswell_6x16` | AOCL-BLAS 5.1 `bli_sgemm_haswell_asm_6x16` (BLIS 0.x interface) | 6 x 16 | `blis/kernel/s_aocl_haswell_6x16.h` | `-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_aocl_haswell_6x16` | tested (Ice Lake, also ASan/UBSan) |
| `openblas_haswell_8x4` | OpenBLAS 0.3.34 `sgemm_kernel_8x4_haswell_2.c` | 8 x 4 (block calls) | `openblas/kernel/s_haswell_8x4.h` | `-mavx -mavx2 -mfma -DULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4` | tested (Ice Lake, also ASan/UBSan) |
| `blis_skx_32x12_l2` | BLIS 2.0 `bli_sgemm_skx_asm_32x12_l2` | 32 x 12 | `blis/kernel/s_skx_32x12_l2.h` | `-mavx -mfma -mavx512f -DULMBLAS_SGEMM_UKERNEL_blis_skx_32x12_l2` | tested (Ice Lake, also ASan/UBSan) |
| `openblas_skylakex_16x4` | OpenBLAS 0.3.34 `sgemm_kernel_16x4_skylakex_3.c` | 16 x 4 (block calls) | `openblas/kernel/s_skylakex_16x4.h` | `-mavx -mfma -mavx512f -DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4` | tested (Ice Lake, also ASan/UBSan) |

Benchmarks, single-threaded, m = n = k = 50 … 4000, n ≥ 1000:

- Intel Xeon Platinum 8358
  ([030](../../../benchmark_results/030-2026-10-28-uc3-icelake-sgemm/README.md)):
  `openblas_skylakex_16x4` 1.007 of native OpenBLAS, 0.925 of oneMKL;
  `blis_skx_32x12_l2` 1.010 of native BLIS skx (same kernel);
  `blis_haswell_6x16` 0.995, `blis_aocl_haswell_6x16` 0.997 of native
  BLIS haswell; `blis_haswell_4x24` drops to 80 … 87 GFLOP/s for n ≥ 2450
  (94 … 96 below).
- Intel Core i5-12500
  ([033](../../../benchmark_results/033-2026-10-28-heim-sgemm/README.md)):
  `openblas_haswell_8x4` 0.992 of native OpenBLAS; `blis_haswell_6x16`
  1.002, `blis_aocl_haswell_6x16` 1.003, `blis_haswell_16x6` 0.991 of
  native BLIS; `blis_haswell_4x24` drops to 105 … 116 GFLOP/s for n > 2000
  (115 … 118 up to 2000).

Tests: configurations `s-NAME` and `s-NAME-small` in `test/config/`.
License: LICENSE sections 7 (BLIS, AOCL-BLAS) and 8 (OpenBLAS).

## SGEMM: AArch64 NEON

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `neon` | ulmBLAS (not derived from another library) | 8 x 12 | `neon/sgemm.tcc`, `neon/ugemm.h` | default on AArch64 (LP64), as for DGEMM | tested (Apple M5: `make check`, also with small block sizes and ASan/UBSan) | [032](../../../benchmark_results/032-2026-10-29-macbook-m5-sgemm/README.md) |

24 accumulators (two per column of the tile), FMLA by element; per k 5
vector loads and 24 FMLA.  C with unit row stride is updated with vector
loads and stores, other strides through a local tile.  MC, KC, NC = 960,
1024, 4080.  n ≥ 1000: 1.12 of OpenBLAS, 1.04 of Eigen (Apple M5).
License: ulmBLAS.

## SGEMM: AArch64 SME2

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `sme` (experimental) | ulmBLAS, after the SME2 DGEMM kernel (tile layout as in `bli_sgemm_armsme_int_2SVLx2SVL` of BLIS PR #921, no code of it) | 32 x 32 | `sme/sgemm.tcc`, `sme/spack.tcc`, `sme/ugemm.h`, `sme/pack.h` | with the SME2 DGEMM kernel (`ARCHFLAGS=-mcpu=apple-m4`) | tested (Apple M5: `make check`, also with small block sizes and ASan/UBSan) | [032](../../../benchmark_results/032-2026-10-29-macbook-m5-sgemm/README.md) |

The tile in ZA0.S … ZA3.S, per k one load of 32 elements of A and of B
and 4 FMOPA; for alpha = 1 and beta = 0 or 1 C is loaded into and stored
from ZA directly, as for DGEMM.  Packing in streaming mode (ZA as
transpose buffer for transposed panels).  MC, KC, NC = 512, 2048, 1024.
n ≥ 1000: 1.023 of Apple Accelerate (1.026 from n = 1450 on; Accelerate
is faster up to n = 1400), 0.86 for n = 50 … 250 (Apple M5).
Requirements as for the SME2 DGEMM kernel.  License: ulmBLAS.

## CGEMM: x86-64

Vendored CGEMM kernels (complex float), selected with exactly one
`-DULMBLAS_CGEMM_UKERNEL_<origin>_<name>` ([`blis/select_c.h`](blis/select_c.h)),
independently of the other types; without one CGEMM uses the reference
kernel.  Details: [blis/README.md](blis/README.md#cgemm-kernels).

| Kernel | Origin | MR x NR | Files | Selection | Status |
|---|---|---|---|---|---|
| `blis_haswell_3x8`, `blis_haswell_8x3` | BLIS 2.0 `bli_cgemm_haswell_asm_3x8`, `_8x3` | 3 x 8, 8 x 3 | `blis/kernel/c_haswell_3x8.h`, `c_haswell_8x3.h` | `-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8` (or `_8x3`) | tested (heim, also ASan/UBSan and valgrind; Ice Lake) |
| `blis_sandybridge_8x4` | BLIS 2.0 `bli_cgemm_sandybridge_asm_8x4` | 8 x 4 | `blis/kernel/c_sandybridge_8x4.h` | `-mavx -DULMBLAS_CGEMM_UKERNEL_blis_sandybridge_8x4` | tested (heim, also ASan/UBSan and valgrind; Ice Lake) |
| `blis_aocl_haswell_3x8` | AOCL-BLAS 5.1 `bli_cgemm_haswell_asm_3x8` (AMD's version, BLIS 0.x interface) | 3 x 8 | `blis/kernel/c_aocl_haswell_3x8.h` | `-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_aocl_haswell_3x8` | tested (heim, also ASan/UBSan and valgrind; Ice Lake) |
| `openblas_haswell_8x2` | OpenBLAS 0.3.34 `cgemm_kernel_8x2_haswell.c` | 8 x 2 (block calls) | `openblas/kernel/c_haswell_8x2.h` | `-mavx -mavx2 -mfma -DULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2` | tested (heim and Ice Lake, also ASan/UBSan; valgrind on heim) |
| `openblas_skylakex_8x2` | OpenBLAS 0.3.34 `cgemm_kernel_8x2_skylakex.c` | 8 x 2 (block calls) | `openblas/kernel/c_skylakex_8x2.h` | `-mavx -mfma -mavx512f -DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2` | tested (Ice Lake, also ASan/UBSan) |

Tests: configurations `c-NAME` and `c-NAME-small` in `test/config/`;
test records in
[investigations/2026-10-gemm-datatypes](../../../investigations/2026-10-gemm-datatypes/README.md#cgemm-x86-64-status).
License: LICENSE sections 7 (BLIS, AOCL-BLAS) and 8 (OpenBLAS).

Benchmarks, single-threaded, m = n = k = 50 … 4000, n ≥ 1000:

- Intel Xeon Platinum 8358
  ([031](../../../benchmark_results/031-2026-10-29-uc3-icelake-cgemm/README.md)):
  `openblas_skylakex_8x2` 0.997 of native OpenBLAS, 0.888 of oneMKL;
  `blis_haswell_3x8` 0.994 of native BLIS haswell, `blis_aocl_haswell_3x8`
  0.996 of native AOCL-BLAS (same kernel).
- Intel Core i5-12500
  ([034](../../../benchmark_results/034-2026-10-29-heim-cgemm/README.md)):
  `openblas_haswell_8x2` 0.992 of native OpenBLAS; `blis_haswell_3x8`
  1.001, `blis_aocl_haswell_3x8` 0.998, `blis_haswell_8x3` 0.992 of native
  BLIS; `blis_sandybridge_8x4` 0.56 of OpenBLAS.

## CGEMM: AArch64 NEON

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `neon` | ulmBLAS (not derived from another library) | 4 x 6 | `neon/cgemm.tcc`, `neon/ugemm.h` | default on AArch64 (LP64), as for DGEMM | tested (Apple M5: `make check`, also with small block sizes and ASan/UBSan) | [035](../../../benchmark_results/035-2026-10-29-macbook-m5-cgemm/README.md) |

As the ZGEMM kernel with two complex elements per register: real and
imaginary accumulators (24), FMLA by element, any strides of C.  MC, KC,
NC = 384, 1024, 4080.  n ≥ 1000: 1.24 of OpenBLAS, 1.14 of Eigen (Apple
M5).  License: ulmBLAS.

## CGEMM: AArch64 SME2

| Kernel | Origin | MR x NR | Files | Selection | Status | Benchmarks |
|---|---|---|---|---|---|---|
| `sme` (experimental) | ulmBLAS (not derived from another library) | 32 x 16 | `sme/cgemm.tcc`, `sme/cpack.tcc`, `sme/ugemm.h`, `sme/pack.h` | with the SME2 DGEMM kernel (`ARCHFLAGS=-mcpu=apple-m4`) | tested (Apple M5: `make check`, also with small block sizes and ASan/UBSan) | [035](../../../benchmark_results/035-2026-10-29-macbook-m5-cgemm/README.md) |

As the SME2 ZGEMM kernel with 32-bit elements: real parts in ZA0.S and
ZA1.S, imaginary parts in ZA2.S and ZA3.S, per k 3 LD2W and 8
FMOPA/FMOPS.  Packing in streaming mode (64-bit elements of ZA as
transpose buffer).  MC, KC, NC = 384, 1024, 1024.  n ≥ 1000: 1.065 of
Apple Accelerate (ahead at all sizes but n = 1050), 1.67 for n = 50 …
250 (Apple M5).  Requirements as for the SME2 DGEMM kernel.  License:
ulmBLAS.
