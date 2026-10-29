# Optimized GEMM micro kernels for float, complex float and complex double (2026-10)

Until now ulmBLAS selects optimized micro kernels for `double` only; SGEMM,
CGEMM and ZGEMM run the reference micro kernel (`ulmblas/level3/ukernel/ref`).
This investigation extends the kernel selection to the other three BLAS
data types, in the order ZGEMM, SGEMM, CGEMM. This file holds the inventory
(phase 0), the plan, and a record of progress for each data type.

*Historical note: the first paragraph and phase 0 describe the state when
the investigation started.  All four types now have optimized kernels
(see the [support matrix](../../README.md#current-micro-kernel-support)).
On the Apple M5 the plan named ports of BLIS kernels for SGEMM; the NEON
and SME2 kernels for float, complex float and complex double were written
for ulmBLAS instead.*

## Phase 0: inventory of the DGEMM kernels and their upstream counterparts

All statements were checked against these upstream trees. The tarballs
were verified with `git get-tar-commit-id` against `git ls-remote` of the tag:

| Tree | Repository | Tag | Commit |
|---|---|---|---|
| BLIS 2.0 | https://github.com/flame/blis | `2.0` | e8566eb3e773fb54d11b33e371d13f22d2941e50 |
| AOCL-BLAS 5.0 | https://github.com/amd/blis | `5.0` | 34d4bbade33a4384cfaff2208c833fc33a311c5d |
| AOCL-BLAS 5.1 | https://github.com/amd/blis | `5.1` | 16f852a065e76e824d77bc39e2baa82ac19ed419 |
| OpenBLAS 0.3.34 | https://github.com/OpenMathLib/OpenBLAS | `v0.3.34` | e0166008be8e466242aa76b2ff75ce3f0fbf574a |
| BLIS PR #921 | https://github.com/Luislo1/blis (head of flame/blis#921) | — | 235ef8ec097c43ab8299c89071af1b5b5e180843 |

### DGEMM kernels in ulmBLAS (master 05745cd)

| Origin | Kernel (selection) | ISA | MR x NR | Packing | License |
|---|---|---|---|---|---|
| ulmBLAS | `ref` (default without SIMD) | C++ | 4 x 4 (type dependent) | ulmBLAS (k-major panels, zero padded) | ulmBLAS |
| ulmBLAS | `sse` (`USE_SSE`) | SSE2 inline asm, written for ulmBLAS (2014) | 4 x 4 | ulmBLAS | ulmBLAS |
| ulmBLAS (port) | `avx` (`USE_AVX`) | AVX2/FMA; hand-expanded port of BLIS `bli_dgemm_haswell_asm_6x8` | 6 x 8 | ulmBLAS | BSD-3 (BLIS) |
| ulmBLAS (port) | `neon` (`USE_NEON`) | NEON; port of BLIS `bli_dgemm_armv8a_asm_8x6r`, operands swapped | 6 x 8 | ulmBLAS | BSD-3 (BLIS) |
| ulmBLAS (port) | `sme` (`USE_SME`) | SME2; derived from BLIS PR #921 `bli_dgemm_armsme_int_4SVLx2SVL` | 32 x 16 | ulmBLAS + streaming-mode packing | BSD-3 (BLIS) |
| BLIS 2.0 | `blis_haswell_6x8`, `_8x6` | AVX2/FMA | 6x8, 8x6 | ulmBLAS (= BLIS layout) | BSD-3 |
| BLIS 2.0 (`old/`) | `blis_haswell_12x4`, `_4x12` | AVX2/FMA | 12x4, 4x12 | ulmBLAS | BSD-3 |
| BLIS 2.0 | `blis_sandybridge_8x4`, `_int_8x4` | AVX | 8 x 4 | ulmBLAS | BSD-3 |
| BLIS 2.0 | `blis_penryn_4x4` | SSE3 | 4 x 4 | ulmBLAS | BSD-3 |
| BLIS 2.0 | `blis_skx_16x14`, `_16x12_l2` | AVX-512F | 16x14, 16x12 | ulmBLAS | BSD-3 |
| AOCL-BLAS 5.0 | `blis_zen4_32x6`, `_8x24` | AVX-512F | 32x6, 8x24 | ulmBLAS | BSD-3 (UT Austin, AMD) |
| AOCL-BLAS 5.1 | `blis_zen5_8x24` | AVX-512F | 8 x 24 | ulmBLAS | BSD-3 |
| OpenBLAS 0.3.34 | `openblas_skylakex_16x12` | AVX-512F | 16 x 12 (B addressing patched) | ulmBLAS | BSD-3 (OpenBLAS) |
| OpenBLAS 0.3.34 | `openblas_haswell_4x8` | AVX2/FMA (`.S`, unchanged) | 4 x 8 (block calls) | ulmBLAS | BSD-3 |

Selection: `-DULMBLAS_DGEMM_UKERNEL_<origin>_<name>` (one at most,
`ulmblas/level3/ukernel/blis/select.h`) replaces the kernel for `double`;
default MC/KC/NC are those of the upstream configuration that uses the
kernel, overridable with `-DULMBLAS_DGEMM_{MC,KC,NC}`. Details are in
`ulmblas/level3/ukernel/blis/README.md` and `.../openblas/README.md`.

### Upstream counterparts for float, complex float, complex double

For each DGEMM kernel, this table lists the kernels that the same upstream
project provides for the other types: in the same source file, or the
kernels that the same upstream configuration registers (`bli_cntx_init_*.c`,
`KERNEL.<TARGET>`). MR x NR are in elements of the type (complex elements
for c, z).

| DGEMM kernel | SGEMM | CGEMM | ZGEMM |
|---|---|---|---|
| ulmBLAS `sse` (own) | — (no upstream) | — | — |
| ulmBLAS `avx` = BLIS haswell 6x8 | BLIS `sgemm_haswell_asm_6x16` | BLIS `cgemm_haswell_asm_3x8` | BLIS `zgemm_haswell_asm_3x4` |
| BLIS `haswell_8x6` | BLIS `sgemm_haswell_asm_16x6` | BLIS `cgemm_haswell_asm_8x3` | BLIS `zgemm_haswell_asm_4x3` |
| BLIS `haswell_12x4`, `4x12` (`old/`) | BLIS `sgemm_haswell_asm_24x4`, `4x24` | none: `bli_cgemm_haswell_asm_` is an empty stub inside `#if 0` | none (same) |
| BLIS `sandybridge_8x4` | BLIS `sgemm_sandybridge_asm_8x8` | BLIS `cgemm_sandybridge_asm_8x4` | BLIS `zgemm_sandybridge_asm_4x4` |
| BLIS `sandybridge_int_8x4` | none (file has the d kernel only) | none | none |
| BLIS `penryn_4x4` | BLIS `sgemm_penryn_asm_8x4` | none (registration commented out, no source) | none |
| BLIS `skx_16x14`, `16x12_l2` | BLIS `sgemm_skx_asm_32x12_l2` | none: the skx configuration registers no complex kernel (complex through BLIS's induced 1m method) | none (same) |
| AOCL `zen4_32x6`, `zen4_8x24`, `zen5_8x24` | the zen4/zen5 configurations use AOCL's copy of `sgemm_skx_asm_32x12_l2` (different from BLIS 2.0's: other source hash) | AOCL's `cgemm_haswell_asm_3x8` (BLIS 0.x interface, modified by AMD) | AOCL `zgemm_zen4_asm_12x4` (zen4, zen5); `zgemm_zen4_asm_4x12` (TRSM block sizes) |
| (AOCL zen, zen2, zen3: BLIS haswell d kernels, not vendored as d) | AOCL `sgemm_haswell_asm_6x16` (modified) | AOCL `cgemm_haswell_asm_3x8` | AOCL `zgemm_haswell_asm_3x4`: modified by AMD (0.x interface, special cases alpha, beta = ±1, 0) |
| OpenBLAS `skylakex_16x12` | `sgemm_kernel_16x4_skylakex_3.c` | `cgemm_kernel_8x2_skylakex.c` | `zgemm_kernel_4x2_skylakex.c` |
| OpenBLAS `haswell_4x8` | `sgemm_kernel_8x4_haswell_2.c` | `cgemm_kernel_8x2_haswell.c` | `zgemm_kernel_4x2_haswell.c` |
| ulmBLAS `neon` = BLIS armv8a 8x6r | BLIS `sgemm_armv8a_asm_12x8r` (firestorm) | none: no armv8a complex kernel in BLIS 2.0 | none |
| ulmBLAS `sme` = BLIS PR #921 | PR #921 `sgemm_armsme_int_2SVLx2SVL` (32 x 32) | none in PR #921 | none |

Facts that matter for the integration:

- **Packing.** BLIS native complex kernels (not the 1m method) and OpenBLAS
  complex kernels read panels of interleaved (re, im) elements, k-major,
  MR (or NR) complex elements per k, which is ulmBLAS's packing for
  `std::complex<T>`. ulmBLAS applies conjugation of A and B while packing,
  so every kernel is called as its no-conjugation variant (OpenBLAS `NN`;
  BLIS kernels have no conjugation parameter).
- **OpenBLAS complex kernels** are block kernels `CNAME(m, n, k, alpha_r,
  alpha_i, A, B, C, ldc)` computing C ← C + alpha·A·B (beta is applied
  before), column-major C. Their B operand is a sequence of 2-column panels
  (`ZGEMM_UNROLL_N` = `CGEMM_UNROLL_N` = 2). With NR = 2, ulmBLAS's packed
  block of B has exactly this layout, so the kernels need no change to
  their assembly (unlike the DGEMM SKYLAKEX kernel). Their register tiles
  are 4 x 6 (HASWELL z), 4 x 12 (SKYLAKEX z), 8 x 6 and 8 x 12 (c), used only
  when a call spans several panels of B. They therefore need the block
  calls of `haswell_4x8`.
- **BLIS 2.0 complex kernels** use the BLIS 1.0 interface. `zgemm_haswell_asm_3x4`
  and `cgemm_haswell_asm_3x8` support only row-major C (`GEMM_UKR_SETUP_CT`,
  row preference). BLIS computes Cᵀ = BᵀAᵀ for column-major C. The 8x6-family
  kernels (`4x3`, `8x3`) and sandybridge prefer columns. The complex kernels
  test beta = 0 with `and(r8b, r9b)`. `and` is an operator name in C++, so the
  vendored code must use the macro header's `AND()`. The d kernels never used it.
- **AOCL kernels** use the BLIS 0.x interface (`dcomplex*`, full tiles),
  AOCL trace macros (`AOCL_DTL_*`) and file-static constant tables.
- **No complex kernels exist for AArch64** in BLIS 2.0 (armv8a) or in BLIS
  PR #921 (SME). The MacBook M5 can therefore only provide reference
  ZGEMM/CGEMM numbers; x86-64 hosts are needed.

### Machines

| Machine | CPU | Kernels it can run |
|---|---|---|
| heim | Intel Core i5-12500 (Alder Lake, AVX2/FMA, no AVX-512) | BLIS haswell, sandybridge, penryn; AOCL haswell; OpenBLAS HASWELL; native OpenBLAS 0.3.34 (HASWELL) and BLIS 2.0 (haswell) in `/tmp/ulmblas-avx.tGdSUq/refs` |
| bwUniCluster 3.0 `cpu_il` | Intel Xeon Platinum 8358 (Ice Lake-SP, AVX-512) | all x86 kernels; OpenBLAS SKYLAKEX, BLIS skx, AOCL, oneMKL natively |
| bwUniCluster 3.0 `cpu` | AMD EPYC 9454 (Zen 4) | all x86 kernels (two Zen 4 jobs of another campaign are queued; not touched) |
| MacBook Air | Apple M5 (NEON, SME2) | AArch64 only: no complex kernels; SGEMM: BLIS armv8a 12x8r, PR #921 SME |

## Plan

1. **Infrastructure (shared by all three types).**
   - Per-type selection macros `ULMBLAS_{S,C,Z}GEMM_UKERNEL_<origin>_<name>`,
     independent of each other and of the DGEMM selection. Each has its own
     select header with MR, NR, default MC/KC/NC of the upstream configuration
     (override `-DULMBLAS_{S,C,Z}GEMM_{MC,KC,NC}`), interface, storage of C,
     row preference, and block calls.
   - One generic adapter for the non-double types (traits plus tag
     dispatch, C++11, no run-time dispatch), sharing the interface classes of
     the DGEMM adapter (BLIS 1.0, BLIS 0.x, OpenBLAS). The DGEMM adapter is
     left unchanged so that DGEMM object code stays identical (checked by
     comparing the objects of default and BLIS-kernel builds).
   - Extend `compat.h` to typed edge-case macros (`ch` = s, d, c, z) with
     `scomplex`/`dcomplex` and the AOCL trace macros.
   - Extend the vendoring scripts with the new kernels (generated files
     record source path, commit and SHA-256; mechanical changes listed).
   - Block calls (`MgemmBlock`) for OpenBLAS kernels of any type.
   - Row-preferring kernels: GEMM computes Cᵀ = BᵀAᵀ when C is column-major,
     as BLIS does (non-double types only).
   - Test configurations per kernel (default and `-small`). In `USE_TESTPARAM`
     builds, a selected kernel fixes MR/NR for its type.
2. **ZGEMM**: BLIS haswell 3x4, 4x3, sandybridge 4x4; AOCL zen4 12x4, 4x12,
   AOCL haswell 3x4; OpenBLAS HASWELL 4x2, SKYLAKEX 4x2. Tests on heim (AVX2
   kernels, with ASan/UBSan and valgrind) and on an Ice Lake compute node
   (AVX-512 kernels). Benchmark n = 50 … 4000 on heim (AVX2 kernels, native
   OpenBLAS/BLIS, reference) and Ice Lake (AVX-512 kernels, OpenBLAS,
   oneMKL, BLIS, AOCL), 8mnk flops.
3. **SGEMM**: BLIS haswell 6x16, 16x6, 24x4, 4x24, sandybridge 8x8, penryn
   8x4, skx 32x12 l2; AOCL skx 32x12 l2 (if different in a way that
   matters), AOCL haswell 6x16; OpenBLAS HASWELL 8x4, SKYLAKEX 16x4; BLIS
   armv8a 12x8r (M5), PR #921 SME 32x32 (M5). Block sizes taken from the
   upstream configuration for float, not from DGEMM. 2mnk flops.
4. **CGEMM**: BLIS haswell 3x8, 8x3, sandybridge 8x4; AOCL haswell 3x8;
   OpenBLAS HASWELL 8x2, SKYLAKEX 8x2. 8mnk flops.

Each type is committed in steps (infrastructure and kernels, tests,
benchmark campaign, documentation) and pushed to master once verified.

## ZGEMM: status

Integrated: the 8 kernels of the plan (table and selection:
[ulmblas/level3/ukernel/README.md](../../ulmblas/level3/ukernel/README.md#zgemm-x86-64),
details: [blis/README.md](../../ulmblas/level3/ukernel/blis/README.md#zgemm-kernels)).
Benchmarks: campaigns
[027](../../benchmark_results/027-2026-10-28-heim-zgemm/README.md) (Intel
Core i5-12500),
[028](../../benchmark_results/028-2026-10-28-uc3-icelake-zgemm/README.md)
(Intel Xeon, Ice Lake) and
[029](../../benchmark_results/029-2026-10-28-macbook-m5-zgemm/README.md)
(Apple M5, the NEON and SME2 kernels).

### Test records (`tests/`)

| File | Machine, compiler | Configurations | Result |
|---|---|---|---|
| `heim-check-z1.txt` | Intel Core i5-12500, GCC 12.2 | the 10 AVX/AVX2 configurations (`z-NAME`, `z-NAME-small`) | 1726 PASS, 0 FAIL |
| `heim-check-z2.txt` | as above, after T0354/T0355 and the checks for complex operands and scalars | the 10 configurations and `testcase01` … `testcase09` | 2930 PASS, 0 FAIL (348 SKIP: NEON, SME configurations) |
| `heim-check-z-san.txt` | as above, `-O1 -fsanitize=address,undefined --param asan-stack=0` | the 10 configurations | 1726 PASS, 0 FAIL, no sanitizer report |
| `heim-valgrind-z.txt` | as above, valgrind 3.19 memcheck, numeric tests | the 10 configurations | 145 PASS each, 0 errors, no leaks |
| `heim-dgemm-object-identity.txt` | as above | `libcblas.a` of master 05745cd and of the branch without a ZGEMM kernel: `-mavx -mfma`, BLIS haswell 6x8, OpenBLAS HASWELL 4x8, `-msse3` | disassembly of all 50 (51) objects identical |
| `icelake-check-z.txt` | Intel Xeon Platinum 8358, GCC 14.2 | the 6 AVX-512 configurations and `testcase06` | 1210 PASS, 0 FAIL |
| `icelake-check-z-san.txt` | as above, with ASan/UBSan | the 6 AVX-512 configurations | 1038 PASS, 0 FAIL, no sanitizer report |
| `icelake-check-z-all.txt` | as above, the tree as committed (on top of master 747b561) | all 16 ZGEMM configurations and `testcase06` | 2930 PASS, 0 FAIL |
| `mac-check-numeric.txt` | Apple M5, Apple clang 21, the tree as committed | `make check-numeric`, `testcase01` … `testcase09` (no vendored kernel) | 1035 PASS, 0 FAIL (298 SKIP: AVX/FMA configurations) |

The AVX-512 kernels were not run under valgrind (valgrind does not
support AVX-512).  Complex test operands have nonzero imaginary parts
(checked in every GEMM test case); alpha and beta include -1, 0, 1, real,
imaginary and complex values.

## CGEMM (x86-64): status

Integrated: the 6 kernels of the plan (table and selection:
[ulmblas/level3/ukernel/README.md](../../ulmblas/level3/ukernel/README.md#cgemm-x86-64),
details: [blis/README.md](../../ulmblas/level3/ukernel/blis/README.md#cgemm-kernels)).
Benchmarks:
[031](../../benchmark_results/031-2026-10-29-uc3-icelake-cgemm/README.md)
(Intel Xeon, Ice Lake) and
[034](../../benchmark_results/034-2026-10-29-heim-cgemm/README.md)
(Intel Core i5-12500).

### Test records (`tests/`)

| File | Machine, compiler | Configurations | Result |
|---|---|---|---|
| `heim-check-c.txt` | Intel Core i5-12500, GCC 12.2 | the 12 configurations (`c-NAME`, `c-NAME-small`; the 2 AVX-512 ones SKIP) | 1736 PASS, 0 FAIL |
| `heim-check-c-san.txt` | as above, `-O2 -fsanitize=address,undefined --param asan-stack=0` | as above | 1736 PASS, 0 FAIL, no sanitizer report |
| `heim-valgrind-c.txt` | as above, valgrind 3.19 memcheck, numeric tests | the 10 AVX/AVX2 configurations | 145 PASS each, 0 errors, no leaks |
| `icelake-check-c.txt` | Intel Xeon Platinum 8358, GCC 14.2 | all 12 configurations | 2082 PASS, 0 FAIL |
| `icelake-check-c-san.txt` | as above, `-O1` with ASan/UBSan | the 4 OpenBLAS configurations | 698 PASS, 0 FAIL, no sanitizer report |

- GCC 14.2 with ASan does not compile the BLIS and AOCL kernels (inline
  assembly: "impossible constraints or there are not enough registers");
  GCC 12.2 does, so their sanitizer run is the one on heim.
- valgrind: the first run of `c-blis_haswell_8x3-small` spun for 3.5 hours
  in T0351 (DGEMM from 8 threads at the same time) under valgrind's default
  scheduler; with `--fair-sched=yes` it passed in 16 minutes.  The plain
  and sanitizer runs of this configuration pass T0351.
