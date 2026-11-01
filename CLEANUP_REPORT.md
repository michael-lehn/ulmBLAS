# GEMM architecture cleanup

**Goal:** historical architectural simplicity + current correctness + current
test coverage.  This is the preparation for a redesign of the GEMM
configuration; no new configuration system is implemented here.

| | |
|---|---|
| Architectural reference | `bb1827b2bc81689196c49c1d7243da7f791285e1` (2026-10-05) |
| Starting point | `6c554eec707e22f118bae5dcaea10ad9945aac54` (GitHub `master`, 2026-11-01) |
| Branch | `cleanup/gemm-restore-simple-architecture` |
| Result | 72 library files removed, 11 changed; about 72 000 lines fewer in `ulmblas/` |
| Tests | 0 failures, no test or configuration of `6c554eec` missing, see [Test results](#4-test-results); native x86-64 **not** run |
| Performance | Apple M5: 1/2 of NEON, 1/14 of SME2 (DGEMM); SSE3 unchanged within noise |

## 1. The historical reference

`bb1827b2` is the last commit before the recent development.  Nothing was
committed between 2026-10-05 and 2026-10-23, so it is exactly the tree as it
stood two weeks ago; its library code dates from 2015.  It is also the merge
base of the two histories described next.

**The history exists in two forms.**  GitHub `master` squashes the development of
10-23 to 10-29 into one commit, `8177d950 Major update`, followed by 15 commits up
to `6c554eec`.  This squash is intentional; it is the current state and the base of
this branch.  The local `master` of the main working tree (`bb07e43c`) still has the
individual commits, which served only to classify the changes in section 2, so
those commit hashes exist only in that clone unless noted.

The main working tree and its `master` were not modified.

## 2. What the recent development changed

Commits of the granular history, plus the changes after the squash.

### Architectural changes (removed)

| Change | Where | Commits |
|---|---|---|
| Interleaved packing of B, chosen by a type check for `double` | `gemm.tcc` | `ac2d97ea`, `ae966281` |
| `UGemmRowPreference`: C^T = B^T A^T for column-major C | `gemm.tcc`, `ugemm.h` | squash |
| `gemm_direct`: small DGEMM without packing, with NEON | `gemmdirect.*`, `gemm.tcc` | `6228fcbc`, `8a9749b0` |
| `MgemmPanel`, `MgemmBlock`: one kernel call per panel or block | `mgemm.tcc` | `968e2507`, `ae966281`, squash |
| `pack_*_partial`: backend hooks for packing partial panels | `gepack.tcc` | `889aa154` |
| Kernel selection for 5 backends, block sizes per kernel (617 lines) | `ugemm.h`, `simd.h`, `blocksize.h`, `level3.h` | `5e0a1595` onwards |
| Alignment per vendored kernel | `memorypool.tcc` | `f530d1a9` |
| Assembly rules for the OpenBLAS HASWELL kernel | `make.inc`, library makefiles | `59120c50` |

### Performance optimizations

| Change | Commits | Decision |
|---|---|---|
| Grouped packing of A (`PACK_A_PANELS`, `PACK_A_GROUP_DISTANCE`) | `0da6b0d0`, `5c674ad4` | removed |
| Tuned block sizes of the removed kernels | `e195beab`, `ad778d48`, `8fe961a3` and others | removed |
| Per-thread `MemoryPool` cache | `04cf17b7` | **kept**, see section 3 |
| Local constants `constexpr` (not an optimization, mechanical) | `d63fd7f0`, `11ee67f6` | **kept**, see section 3 |

### New micro kernels (all removed)

| Family | Kernels |
|---|---|
| ulmBLAS own or derived | AVX/FMA (double, 6x8); NEON (d 6x8, s 8x12, c 4x6, z 4x3); SME2 (d 32x16, s 32x32, c 32x16, z 16x16) and its packing |
| BLIS, double | `penryn_4x4`, `sandybridge_8x4`, `sandybridge_int_8x4`, `haswell_6x8`, `8x6`, `12x4`, `4x12`, `skx_16x14`, `skx_16x12_l2` |
| AOCL-BLAS, double | `zen4_32x6`, `zen4_8x24`, `zen5_8x24` |
| OpenBLAS, double | `skylakex_16x12`, `haswell_4x8` |
| Vendored, other types | float, complex float and complex double kernels from BLIS, AOCL-BLAS and OpenBLAS (24 files) |

### Correctness fixes (all retained)

| Fix | Issue | Files |
|---|---|---|
| SSE GEMM kernel read C for beta = 0; missing `memory` clobbers | #0008, #0009 | `ukernel/sse/*` |
| SSE labels not local on Mach-O | #0007 | `ukernel/sse/ugemm.tcc` |
| SSE kernels only on x86-64; TRSM block size checked at compile time | | `simd.h`, `ukernel/sse/utr*.h` |
| `alignas` for arrays accessed with aligned SSE loads | | `ukernel/utr*.tcc`, level 1 kernels |
| `gescal`, `trlscal`, `truscal`: NaN and Inf kept for alpha = 0 with general strides | #0006, #0010 | `level1extensions/` |
| `msylrk`, `msyurk` for MR, NR that do not divide each other | | `mkernel/` |
| `static_assert` on block sizes, otherwise the packing buffers overflow | | `blocksize.h` |
| Out of memory: message and `abort`, no exception through the interfaces | | `memorypool.*`, `lbt/Makefile` |
| `dgetrf` row major: wrong results, writes outside the matrix | #0003, #0004 | `interfaces/lapack/C` |
| Computation continued after `xerbla`; XERBLA name length | #0005, #0012 | interfaces |
| Offsets overflowed `int`; `ULMBLAS_INDEX_TYPE` | #0015 | interfaces, level 1 to 3 |
| `rotg`, `nrm2`, `i?amax` with NaN, alpha = 0 conventions, missing level 1 routines | #0001, #0002, #0011, #0013, #0014 | `level1/`, `level2/` |
| Single precision scalars in the F77 level 3 interface | | `interfaces/blas/F77` |
| Compilation on non-x86 targets | | `level3.h`, `level1extensions.h`, `fusefactor.h` |

### Needed by the current tests and benchmarks

| Dependency | Decision |
|---|---|
| `make check`, numeric tests, Netlib, `make check-reference-lapack`, examples | kept |
| `USE_TESTPARAM_KEEP_KERNELS` | kept, `testcase12` and the configurations of removed kernels use it |
| `MemoryPool` behavior (T0350, T0353, T0357, T0358) | kept, source and tests identical to `6c554eec` |
| Fault injection through `posix_memalign` (T0358) | kept, needs the current allocator |
| Public `ulmblas_?gemm_strided`, `make lbt` | kept |
| `make bench-gemm`, `bench/gemm/ulmblas-kernels.cc`, `make check_arch` | kept, reduced to the remaining kernels |
| `benchmark_results/`, `investigations/` | kept; links to removed READMEs now point to `6c554eec` |
| Test configurations of removed kernels (81 files) | kept, reported as SKIP, see section 4 |
| `mk/vendor_*.py` (generators of vendored kernels, not tests) | removed with the kernels; at `6c554eec` |

## 3. The result

`gemm.tcc`, `mkernel/mgemm.tcc` and `pack/gepack.tcc` are the text of
`bb1827b2`; the only differences are `constexpr` block sizes (5, 2 and 4 lines).
`ugemm.h` additionally includes `config/simd.h`.  The micro kernels are again
`ref` and `sse`.

Not restored, deliberately:

- **`MemoryPool`.**  `memorypool.tcc` is byte-identical to `6c554eec`.  The pool of
  `bb1827b2` allocates with `new T[]` and through `std::list` and
  `std::unordered_map`.  These can throw through the C and Fortran interfaces,
  which `6c554eec` fixed and `make lbt` rejects at link time.  The alignment
  branches for the AVX and vendored kernels are inactive until those kernels
  return.
- **`LICENSE`.**  Byte-identical to `6c554eec`, because the removed kernels and
  their license notices will return.
- **`constexpr`.**  Mechanical, not architecture-specific, used throughout; going
  back would touch more than 150 definitions without any benefit.
- **Everything outside GEMM.**  `ukernel/sse`, `ref`, level 1 and 2, `lapack`,
  the sources of the interfaces and all other macro kernels are unchanged from
  `6c554eec`, which is why the fixes above are retained.

Tests: `test/numeric` is byte-identical to `6c554eec`, and all of its test
configurations are present.  `test/Makefile.config` and `test/run-tests.sh` gained
the explicit SKIP and its summary, described in section 4.

Documentation: `INSTALL.md`, `BUILDING.md`, `ARCHITECTURE.md`,
`bench/gemm/README.md` and `ulmblas/level3/ukernel/README.md` describe the
remaining kernels.  `README.md`, `PERFORMANCE.md` and `M5_EXPERIMENTS.md` keep
their text and results below a note that they describe the state before the
cleanup; I did not rewrite the prose, which will apply again once kernels return.

## 4. Test results

Final commit, Apple M5, macOS 26, Apple clang 21, gfortran 16.2.  **No test has
disappeared:** every (configuration, test ID) result of `6c554eec` also exists on
this branch, in all 88 configurations and on both architectures.  The comparison
is `investigations/2026-11-gemm-cleanup/test-coverage.txt`, produced by
`compare-test-results.py` in the same directory.

### Correction of the first version of this cleanup

The first version lost coverage.  It reported 184 fewer tests as harmless, but:

- it deleted `testcase08` (NEON, small blocks) and 77 optional configurations of
  vendored kernels;
- it reused `testcase06`, `07` (AVX/FMA) and `09` (SME2) for other purposes, so
  their kernel coverage vanished with the counts unchanged.  That is why comparing
  counts showed only 184 tests, although 368 arm64 results had lost their kernel;
- it edited `test/numeric` and the alignment logic of `memorypool.tcc`.

All of this is now restored.  The new configurations moved to new numbers.

### Default `make check`

| Run | PASS | FAIL | SKIP |
|---|---:|---:|---:|
| arm64 `make check`, `6c554eec` | 1638 | 0 | 392 |
| arm64 `make check`, this branch | 1454 | 0 | 1128 |
| x86-64 under Rosetta 2, `make check-numeric`, `6c554eec` | 1119 | 0 | 636 |
| x86-64 under Rosetta 2, `make check-numeric`, this branch | 1437 | 0 | 795 |

A configuration has 184 tests on arm64 (156 numeric, 2 header, 1 example, 25
Netlib) and 159 without Netlib; the 6 registry tests run once.

What happened to each arm64 result of `6c554eec`:

| Before -> now | Results | Configurations | Meaning |
|---|---:|---|---|
| PASS -> PASS | 1270 | registry, `01`-`05`, `10`, `11` | still executed |
| PASS -> SKIP, kernel removed | 368 | `08` (NEON), `09` (SME2) | backend temporarily removed |
| SKIP -> SKIP, CPU or compiler | 368 | `06`, `07` (AVX/FMA) | backend impossible on arm64, as before |
| SKIP -> SKIP, 64-bit index | 24 | `10`, `11` | Netlib CBLAS programs pass `int`, as before |
| **disappeared** | **0** | | |

New on arm64: `testcase13` adds 184 PASS; `12` and `14` add 368 SKIP because they
need x86.  **Why the PASS count is 184 lower:** `08` and `09` lost 368 and `13`
added 184.

Under Rosetta 2: 1119 PASS -> PASS, 636 PASS -> SKIP (kernel removed, the four
SSE3 configurations `blis-penryn_4x4`, `s-blis_penryn_8x4` and their `-small`
variants, which are optional and not in the first table; that it equals the 636
SKIPs of the table is a coincidence), 12 243 SKIP -> SKIP
(no AVX, NEON or SME), 0 disappeared.  `testcase12` and `13` add 318 PASS.

### The test configurations

| Configuration | Tests | Status here |
|---|---|---|
| `testcase01`-`03`, `05`, `11` | reference kernel, small blocks | run |
| `testcase04`, `10` | default kernel | run; **arm64: now reference instead of NEON**, x86-64: SSE as before |
| `testcase06`, `07` | AVX/FMA kernel | SKIP: kernel removed; no AVX here |
| `testcase08` | NEON kernel, small blocks | SKIP: kernel removed |
| `testcase09` | SME2 kernel, small blocks | SKIP: kernel removed |
| 77 optional vendored configurations | BLIS, AOCL-BLAS, OpenBLAS | SKIP with `CONFIGS=...`; as on `6c554eec`, not part of the default `make check` |
| `testcase12` (new) | SSE kernels, small blocks | was `07` in the first version; x86-64 only; no configuration ran them across block boundaries before |
| `testcase13` (new) | reference kernel, MR = 32, NR = 16 | was `09` in the first version; nearly all tiles partial |
| `testcase14` (new) | SSE kernels in a `-mavx -mfma` build | was `06` in the first version; needs AVX |

**Mechanism.**  The 81 configurations of removed kernels are the files of
`6c554eec` plus one line, for example `CONFIG_KERNEL_REMOVED = NEON`.  The tests stay
registered and are reported one by one as SKIP.  The CPU probe comes first: the
reason "micro kernel temporarily removed" appears only where the test would run
if the kernel were present, so it never hides a backend the machine cannot run.
`make check` now ends with SKIP counts by configuration and reason.  When a
kernel returns, delete that line from its configurations.  Whether they pass then
depends on how the kernel is reintroduced: nothing here has compiled them.

### Are all generic numerical tests still executed?

Yes.  All 156 numeric test IDs run in each of the 8 configurations that run on
arm64 (`01`-`05`, `10`, `11`, `13`), and their sources are unchanged.  All 25 Netlib
programs run in 6 of them; `10` and `11` run the 13 Fortran programs, as before.
Two limits:

- **A PASS no longer exercises what some tests were written for.**  T0334, T0348
  (SME2 packing), T0352 (direct path), T0353 (32-byte and vendored alignment) and
  T0355 (vendored kernels for other types) still run and pass, but those code paths
  do not exist here.  They become meaningful again when the kernels return.
- **Silent backend change on arm64.**  `testcase04` and `10` are 368 PASS -> PASS
  results that tested NEON at default block sizes and now test the reference
  kernel, so more NEON coverage is missing than the SKIPs of `testcase08` show.
  They return to NEON automatically once the kernel returns.

### Other checks of the final commit

| Run | PASS | FAIL | SKIP | Note |
|---|---:|---:|---:|---|
| `make check-numeric` with g++-16, arm64 | 642 | 0 | 159 | `01`, `04`, `05`, `13`; `09` skipped |
| ASan + UBSan, arm64 | 642 | 0 | 0 | `01`, `04`, `05`, `13`, with `ASAN_OPTIONS=allocator_may_return_null=1` |
| UBSan, x86-64 under Rosetta 2 | 324 | 0 | 0 | `04`, `12` (SSE kernels) |
| `make check-reference-lapack` | 22 | 0 | | and 2 REF: failures identical to the reference BLAS |

- Without `allocator_may_return_null=1`, ASan aborts on the 2^57-byte request of
  T0357 before ulmBLAS can handle it.  The unmodified `6c554eec` fails the same way.
- The ASan run over all configurations (1270 PASS, the 8 T0357 aborts above) was
  made on the first version and not repeated.
- `make`, `f77blas`, `atlblas`, `atllapack`, `lbt` (including its no-throw link check)
  and `check_arch` succeed, and `make check_benchmark_results` reports no problems.

| Requested attention | Covered by |
|---|---|
| Arbitrary row and column strides | T0800-T0805 (including negative and zero strides of A and B), T0342, T0347, T0348 |
| Row and column major | T0300-T0307, T0401-T0407, Netlib CBLAS, Reference-LAPACK CBLAS |
| Partial tiles and boundaries | T0334, T0341, T0342, T0352, T0355; small block configurations `01`-`03`, `05`, `11`-`13` |
| `alpha`, `beta`, `beta = 0` | T0343-T0346, T0354, T0600, T0805 |
| Fixed numerical and memory-safety bugs | regression tests in `test/known_issues.md`; T0351, T0356-T0358 |

## 5. Benchmarks

**Conditions:** one MacBook Air (Apple M5), mains power, single thread, not
isolated from other load.  Two interleaved rounds, the better one shown.
**Rounds differ by 15 to 30%**, so treat differences below about 30% as noise.
This is a baseline, not a benchmark campaign.  x86-64 ran under Rosetta 2, which
says little about native SSE3 speed but does allow a before and after comparison.
Raw data and the driver for non-square shapes:
[investigations/2026-11-gemm-cleanup](investigations/2026-11-gemm-cleanup/results.txt).

### Square GEMM on arm64, GFLOP/s (m = n = k)

| n | DGEMM SME2 | DGEMM NEON | **DGEMM after** | SGEMM NEON | **SGEMM after** | ZGEMM NEON | **ZGEMM after** | CGEMM NEON | **CGEMM after** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 35.7 | 32.7 | **10.2** | 10.4 | **8.4** | 21.7 | **10.9** | 30.4 | **11.1** |
| 32 | 165.8 | 35.5 | **26.8** | 58.4 | **15.5** | 44.4 | **13.0** | 78.5 | **13.5** |
| 128 | 337.6 | 52.4 | **29.3** | 102.5 | **17.5** | 56.5 | **13.4** | 112.1 | **14.0** |
| 512 | 423.2 | 59.1 | **30.0** | 121.9 | **18.0** | 61.4 | **13.5** | 123.9 | **14.1** |
| 1024 | 428.8 | 60.1 | **30.7** | 124.5 | **18.1** | 61.6 | **13.5** | 124.3 | **14.1** |
| 4000 | 433.2 | 61.5 | **30.8** | 126.3 | **18.1** | 60.7 | **13.0** | 124.3 | **12.7** |

| Regression at n = 4000 | DGEMM | SGEMM | ZGEMM | CGEMM |
|---|---:|---:|---:|---:|
| against NEON | 2.0 x | 7.0 x | 4.7 x | 9.8 x |
| against SME2 | 14 x | 88 x | 32 x | 117 x |

For comparison, Apple Accelerate is a median of 10.9 times as fast as the
cleaned-up DGEMM over n = 8 to 4000.

### DGEMM on x86-64 (SSE3), under Rosetta 2, GFLOP/s

| n | before | after | | shape m x n x k | before | after |
|---:|---:|---:|---|---|---:|---:|
| 8 | 9.1 | 8.6 | | 8 x 8 x 8 | 8.9 | 8.2 |
| 32 | 22.7 | 22.9 | | 16 x 16 x 16 | 16.7 | 16.7 |
| 128 | 26.5 | 26.6 | | 256 x 256 x 256 | 26.2 | 27.4 |
| 512 | 25.4 | 27.2 | | **2048 x 64 x 2048** | 24.5 | **22.0** |
| 1024 | 27.2 | 26.0 | | 64 x 2048 x 2048 | 25.7 | 26.5 |
| 4000 | 26.8 | 27.7 | | 2048 x 2048 x 64 | 25.5 | 25.8 |

**No regression is distinguishable from noise** on the only optimized path that
remains.  The 10% at 2048 x 64 x 2048 is the shape where the removed grouped
packing is documented to help (2 to 13% on an Intel Core i5-12500), so it may be
real, but this measurement cannot show it.

### Non-square and small DGEMM on arm64, GFLOP/s

| Shape m x n x k | SME2 | NEON | **after** |
|---|---:|---:|---:|
| 8 x 8 x 8 | 35.4 | 34.1 | **9.5** |
| 16 x 16 x 16 | 46.3 | 44.9 | **19.7** |
| 2048 x 64 x 2048 | 260.0 | 43.5 | **26.2** |
| 64 x 2048 x 2048 | 238.0 | 45.9 | **28.3** |
| 2048 x 2048 x 64 | 238.6 | 54.5 | **29.2** |

Small problems lose about 3.5 times at n = 8: the direct path is gone.

### Observations

- The reference kernels are compiled C++, so their speed depends on the compiler
  vectorizing them.  SGEMM (18 GFLOP/s) is **slower** than DGEMM (31) with the
  reference kernels, the opposite of what vector width suggests.
- SGEMM at n = 8 is faster after the cleanup than with SME2 (8.4 against 4.9),
  and CGEMM and ZGEMM are about equal there: entering streaming mode costs more
  than such a small problem gains.

## 6. Unresolved problems

1. **Native x86-64 is untested.**  The SSE kernels are now the only assembly, and
   they ran only under Rosetta 2, without Netlib (no x86-64 gfortran), without
   `testcase14` (no AVX) and without GCC on x86, the compiler that exposed the
   missing clobbers of #0009.  Run on a native machine: `make -j8 check`, and
   with `CXX=g++ CC=gcc FC=gfortran`.
2. **Coverage of the removed kernels is registered, not exercised,** until they
   return (section 4).
3. **Minor leftovers:** `make check_arch` chooses its rows by `uname`, so a cross
   build for x86-64 on arm64 labels the SSE kernel "reference kernels" (the kernel
   it reports is correct).  Its rows and `bench/gemm/ulmblas-kernels.cc` list only
   the remaining kernels; these are tools, not tests.
