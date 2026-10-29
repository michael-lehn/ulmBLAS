# Test records of the BLIS- and OpenBLAS-derived DGEMM micro kernels

`make check` (numeric tests, Netlib BLAS and CBLAS test programs) for the
test configurations `blis-NAME` (default block sizes) and `blis-NAME-small`
(small block sizes, see `test/config/`), on native hardware.  Each file is
the summary that `make check` prints.

| File | Machine, compiler | Revision | Configurations | Result |
|---|---|---|---|---|
| `check-all-summary.txt` | heim: Intel Core i5-12500 (AVX2/FMA), GCC 12.2, gfortran 12.2 | 4632dcb | all default configurations `testcase01` … `testcase09` and the haswell_6x8, haswell_8x6, sandybridge_8x4, penryn_4x4 configurations | 2268 PASS, **8 FAIL** (all `sandybridge_8x4`: segmentation faults, misaligned packing buffers), 344 SKIP (NEON, SME configurations) |
| `check-sb-summary.txt` | heim, as above | f530d1a (packing buffers aligned to the kernel) | sandybridge_8x4, haswell_8x6 | 686 PASS, 0 FAIL |
| `check-san-summary.txt` | heim, as above, `-O1 -fsanitize=address,undefined --param asan-stack=0` | f530d1a | haswell_6x8, haswell_8x6, sandybridge_8x4, penryn_4x4, `testcase06` | 1536 PASS, 0 FAIL, no sanitizer reports |
| `icelake-check-f530d1a.txt` | bwUniCluster 3.0 `cpu_il`: Intel Xeon Platinum 8358 (Ice Lake, AVX-512), GCC 14.2 | f530d1a | `testcase06` and all 9 kernels (18 configurations) | 3202 PASS, **34 FAIL** (all `zen5_8x24`, see below) |
| `zen4-check-f530d1a.txt` | bwUniCluster 3.0 `cpu`: AMD EPYC 9454 (Zen 4, AVX-512), GCC 14.2 | f530d1a | `testcase06` and all 9 kernels (18 configurations) | 3202 PASS, **34 FAIL** (all `zen5_8x24`, see below) |
| `icelake-check-43d6e1f.txt`, `zen4-check-43d6e1f.txt` | both AVX-512 machines, as above | 43d6e1f (`zen5_8x24` of AOCL-BLAS 5.0 with row-major C only, through a local tile) | as above | 3236 PASS, 0 FAIL on both |
| `icelake-check-f2cc04f.txt`, `zen4-check-f2cc04f.txt` | both AVX-512 machines, as above | f2cc04f (`zen5_8x24` from AOCL-BLAS 5.1, any storage of C) | as above | 3236 PASS, 0 FAIL on both |
| `icelake-check-6aff9ca.txt`, `zen4-check-6aff9ca.txt` | both AVX-512 machines, as above | 6aff9ca (adds the OpenBLAS-derived kernel `openblas-skylakex_16x12`; the code is identical in c0d9c06) | as above and the two OpenBLAS configurations (20 configurations) | 3576 PASS, 0 FAIL on both |
| `heim-check-old-haswell.txt`, `heim-check-san-old-haswell.txt` | heim, GCC 12.2, as above (also `-O1 -fsanitize=address,undefined --param asan-stack=0`) | working tree after ed0f3c5 (the commit that adds the kernels) | `haswell_12x4`, `haswell_4x12` (4 configurations) | 686 PASS, 0 FAIL; with sanitizers 686 PASS, 0 FAIL, no reports |
| `icelake-check-ac2d97e.txt`, `zen4-check-ac2d97e.txt` | both AVX-512 machines | ac2d97e (B packed interleaved for the OpenBLAS kernel) | all 25 kernel configurations and `testcase06` | 4596 PASS, 0 FAIL on both |
| `heim-check-interleave-forced.txt` | Intel Core i5-12500 | ac2d97e with the interleaved packing forced on for every kernel and type (test only; under Rosetta 441 PASS without the Netlib suites) | `testcase06`, `blis-haswell_6x8`, `-small` | 516 PASS, 0 FAIL |
| `icelake-check-e15039f.txt`, `zen4-check-e15039f.txt` | both AVX-512 machines | e15039f (adds `sandybridge_int_8x4`) | all 25 kernel configurations and `testcase06` | 4596 PASS, 0 FAIL on both |
| `heim-check-sandybridge_int.txt` | Intel Core i5-12500 | working tree before the commit of `sandybridge_int_8x4` | `blis-sandybridge_int_8x4`, `-small`, also with ASan/UBSan | 346 PASS, 0 FAIL, no sanitizer report; the first run failed (8 FAIL, aligned stores to C, see below) |
| `icelake-check-782af7e.txt`, `zen4-check-782af7e.txt` | both AVX-512 machines | 782af7e (adds `haswell_12x4`, `haswell_4x12`) | all 23 kernel configurations and `testcase06` | 4256 PASS, 0 FAIL on both |
| `icelake-check-968e250.txt`, `zen4-check-968e250.txt` | both AVX-512 machines | 968e250 (OpenBLAS kernel called once per panel of B) | as above | 4256 PASS, 0 FAIL on both |

Notes:

- The sandybridge_8x4 failures of `check-all-summary.txt` were fixed by
  f530d1a: with `-mavx` and without FMA ulmBLAS uses its SSE kernel set,
  and the packing buffers were only 16-byte aligned; the kernel loads B
  with `vmovapd` (32 bytes).  Rosetta 2 (macOS), where all tests had
  passed before, does not fault on such loads.
- AddressSanitizer: the stack instrumentation is disabled
  (`--param asan-stack=0`).  With it GCC cannot compile the kernels
  ("'asm' operand has impossible constraints"): the assembly clobbers 14 of
  the 16 general purpose registers.  Heap accesses (all matrices and packed
  panels), globals and leaks are checked, as is everything UBSan checks.
- `zen5_8x24` (AOCL-BLAS 5.0, `bli_dgemm_avx512_asm_8x24`) fails every test
  with beta = 0 when C is column major or has general strides: the
  kernel tests beta == 0 with `vcomisd xmm1, xmm2`, but only the code path
  for row-major C clears `zmm2` before; in the column-major and general
  stride paths `zmm2` still holds elements of B, so the result of the test
  depends on the data (C is read for beta = 0, giving NaN for NaN or
  uninitialized C; in principle a nonzero beta equal to that element of B
  would be treated as zero).  ulmBLAS stores C column major.  This is a
  defect of the upstream kernel, not of the integration; the kernel is not
  usable in ulmBLAS until the adapter avoids these paths.  Both remedies
  pass all tests: 43d6e1f lets the 5.0 kernel write row-major C only
  (the only path that clears `zmm2`); f2cc04f vendors the kernel from
  AOCL-BLAS 5.1, where AMD added `vxorpd zmm2, zmm2, zmm2` before all store
  paths, and is the version in use.

### `sandybridge_int_8x4`: aligned stores to C

The first native run on heim crashed in the dgemm tests (segmentation
fault; under Rosetta all 296 tests had passed, because Rosetta does not
fault on misaligned `vmovapd`).  UBSan located it: the kernel stores C
with `_mm256_store_pd`, which requires 32-byte aligned addresses, although
it is written for arbitrary C (`GEMM_UKR_SETUP_CT`, not
`GEMM_UKR_SETUP_CT_ALIGNED`): in BLIS, C passes through an aligned
temporary tile when it is not aligned.  ulmBLAS's adapter now has the
storage class 4 (`select.h`): unless C is column major with 32-byte
aligned columns, the kernel computes into a 64-byte aligned local tile.
