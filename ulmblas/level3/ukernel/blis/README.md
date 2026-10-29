# BLIS-derived GEMM micro kernels

This directory contains DGEMM, SGEMM, CGEMM and ZGEMM micro kernels of
[BLIS](https://github.com/flame/blis) and of AMD's BLIS fork
[AOCL-BLAS](https://github.com/amd/blis) for x86-64, and the code that lets
ulmBLAS call them (also for the OpenBLAS kernels of `../openblas`).  The
sections up to [Tests](#tests) describe the DGEMM kernels; the
[ZGEMM](#zgemm-kernels), [SGEMM](#sgemm-kernels) and
[CGEMM](#cgemm-kernels) kernels have their own sections.  They are **not
original ulmBLAS kernels**: the assembly code is that of BLIS and
AOCL-BLAS, vendored unchanged; ulmBLAS provides the packing, the blocking
(macro kernel, loops around it) and an adapter.

## Status

Tested with `make check` (numeric tests, Netlib BLAS and CBLAS test
programs) on native hardware; test records in
[`investigations/2026-10-bwunicluster3/kernels/tests`](../../../../investigations/2026-10-bwunicluster3/kernels/tests/README.md):

| Kernel | Intel Core i5-12500 (AVX2), GCC 12.2 | Intel Xeon Platinum 8358 (Ice Lake, AVX-512), GCC 14.2 | AMD EPYC 9454 (Zen 4, AVX-512), GCC 14.2 | Status |
|---|---|---|---|---|
| `haswell_6x8` | pass, also with ASan/UBSan | pass | pass | verified |
| `haswell_8x6` | pass, also with ASan/UBSan | pass | pass | verified |
| `haswell_12x4` | pass, also with ASan/UBSan | pass | pass | verified; BLIS 0.x kernel from `old/` |
| `haswell_4x12` | pass, also with ASan/UBSan | pass | pass | verified; BLIS 0.x kernel from `old/` |
| `sandybridge_8x4` | pass, also with ASan/UBSan | pass | pass | verified |
| `sandybridge_int_8x4` | pass, also with ASan/UBSan | pass | pass | verified; intrinsics version of `sandybridge_8x4`, aligned stores to C (storage class 4) |
| `penryn_4x4` | pass, also with ASan/UBSan | pass | pass | verified |
| `skx_16x14` | (no AVX-512) | pass | pass | verified (no sanitizer run yet) |
| `skx_16x12_l2` | (no AVX-512) | pass | pass | verified (no sanitizer run yet); not used by any BLIS configuration |
| `zen4_32x6` | (no AVX-512) | pass | pass | verified (no sanitizer run yet) |
| `zen4_8x24` | (no AVX-512) | pass | pass | verified (no sanitizer run yet) |
| `zen5_8x24` | (no AVX-512) | pass | pass | verified (no sanitizer run yet); from AOCL-BLAS 5.1, see below |

`zen5_8x24` was first vendored from AOCL-BLAS 5.0, where it gave wrong
results for beta = 0 when C is column major or has general strides (the
storage ulmBLAS uses), on both AVX-512 CPUs: in these code paths the 5.0
kernel compares beta with `xmm2`, which only its row-major path clears;
otherwise `zmm2` still holds elements of B.  AOCL-BLAS 5.1 fixed this
upstream (`vxorpd zmm2, zmm2, zmm2` before all store paths), and the
kernel is now vendored from 5.1.  The adapter was not at fault: it passes
the arguments of the kernel's interface unchanged.

## Performance

Inside ulmBLAS, with the block sizes of the BLIS configuration that uses
them, the kernels reach the performance of native BLIS with the same
kernel for n = m = k ≥ 1000:

| CPU | Kernels | Inside ulmBLAS / native BLIS (median per kernel) | Campaign |
|---|---|---|---|
| Intel Core i5-12500 | haswell 6x8, 8x6, sandybridge 8x4, penryn 4x4 | 1.000-1.015 | [011-2026-10-26-heim-blis-kernels](../../../../benchmark_results/011-2026-10-26-heim-blis-kernels/README.md), MC iteration [013-2026-10-27-heim-blis-kernels-mc](../../../../benchmark_results/013-2026-10-27-heim-blis-kernels-mc/README.md) |
| Intel Xeon Platinum 8358 | skx 16x14, 16x12 l2, haswell 6x8, 8x6, sandybridge 8x4, penryn 4x4 | 0.963-1.007 (skx 16x14: 1.007) | [012-2026-10-26-uc3-icelake-blis-kernels](../../../../benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md) |

On the Intel Xeon Platinum 8358 the AOCL kernel `zen4_32x6` inside ulmBLAS
is the fastest of these kernels: 90-95 GFLOP/s, 0.96 of OpenBLAS.

## Selection

Exactly one of them can be selected at build time; it replaces the micro
kernel for `double` (all other data types keep the kernel that ulmBLAS
selects without it).  Without a selection nothing changes: ulmBLAS uses its
own kernels (reference, SSE, the AVX/FMA kernel in `../avx`, NEON, SME2).

## Selecting a kernel

```sh
make cblas ARCHFLAGS="-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_skx_16x14"
```

`ARCHFLAGS` must enable the instruction set of the kernel (the build stops
with an error otherwise), and the CPU must support it at run time.  The
cache block sizes default to the values of the BLIS configuration that uses
the kernel and can be changed with `-DULMBLAS_DGEMM_MC=…`,
`-DULMBLAS_DGEMM_KC=…`, `-DULMBLAS_DGEMM_NC=…` (MC must be a multiple of MR
and NR, NC a multiple of NR).

| Name (`ULMBLAS_DGEMM_UKERNEL_blis_…`) | Upstream function | Source | ISA (`ARCHFLAGS`) | MR x NR | Default MC/KC/NC (BLIS configuration) | Storage of C in assembly |
|---|---|---|---|---|---|---|
| `haswell_6x8` | `bli_dgemm_haswell_asm_6x8` | BLIS 2.0 | AVX, FMA (`-mavx -mfma`) | 6 x 8 | 72/256/4080 (haswell) | row or column major |
| `haswell_8x6` | `bli_dgemm_haswell_asm_8x6` | BLIS 2.0 | AVX, FMA | 8 x 6 | 72/256/4080 (haswell, column preferring) | row or column major |
| `haswell_12x4` | `bli_dgemm_haswell_asm_12x4` | BLIS 2.0, `kernels/haswell/3/old/` | AVX, FMA | 12 x 4 | 72/256/4080 (no BLIS configuration uses it: haswell values) | any (column major or general strides) |
| `haswell_4x12` | `bli_dgemm_haswell_asm_4x12` | BLIS 2.0, `kernels/haswell/3/old/` | AVX, FMA | 4 x 12 | 72/256/4080 (as `haswell_12x4`) | any (row major or general strides) |
| `sandybridge_8x4` | `bli_dgemm_sandybridge_asm_8x4` | BLIS 2.0 | AVX (`-mavx`), no FMA | 8 x 4 | 96/256/4096 (sandybridge) | column major |
| `sandybridge_int_8x4` | `bli_dgemm_sandybridge_int_8x4` | BLIS 2.0 (C with intrinsics) | AVX (`-mavx`), no FMA | 8 x 4 | 96/256/4096 (as `sandybridge_8x4`; no configuration uses it) | column major, 32-byte aligned columns |
| `penryn_4x4` | `bli_dgemm_penryn_asm_4x4` | BLIS 2.0 | SSE3 (`-msse3`) | 4 x 4 | 384/384/4096 (penryn) | column major, 16-byte aligned |
| `skx_16x14` | `bli_dgemm_skx_asm_16x14` | BLIS 2.0 | AVX-512F (`-mavx -mfma -mavx512f`) | 16 x 14 | 224/256/3752 (skx: MC 240) | column major |
| `skx_16x12_l2` | `bli_dgemm_skx_asm_16x12_l2` | BLIS 2.0 | AVX-512F | 16 x 12 | 240/256/3744 (no BLIS configuration uses it) | column major |
| `zen4_32x6` | `bli_dgemm_zen4_asm_32x6` | AOCL-BLAS 5.0 | AVX-512F | 32 x 6 | 96/512/4002 (zen4 Genoa: MC 128) | any (gather/scatter) |
| `zen4_8x24` | `bli_dgemm_zen4_asm_8x24` | AOCL-BLAS 5.0 | AVX-512F | 8 x 24 | 120/512/4008 (zen4, block sizes of TRSM) | any |
| `zen5_8x24` | `bli_dgemm_avx512_asm_8x24` | AOCL-BLAS 5.1 | AVX-512F | 8 x 24 | 120/512/2016 (zen5) | any |

Where the BLIS value of MC is not a multiple of both MR and NR (a
condition of ulmBLAS, whose symmetric and triangular routines use MC x MC
blocks), the next smaller multiple is used.

`haswell_6x8` is the same assembly code as ulmBLAS's own AVX/FMA kernel
(`../avx/ugemm.tcc`, a hand-expanded port of it); it is included to compare
the vendored code path with that port.  The kernels `haswell_12x4` and `haswell_4x12` are BLIS 0.x kernels from
`kernels/haswell/3/old/`, which BLIS 2.0 no longer builds: raw inline
assembly with named labels (`.DLOOPKITER`), which the vendoring script
replaces by local, unique labels (`ULMBLAS_BLIS_LABEL "DLOOPKITER%=`, i.e.
`.L…%=` on ELF and `L…%=` on Mach-O); the code is otherwise unchanged.
`sandybridge_int_8x4` is written in C with AVX intrinsics; the script adds
explicit casts to its 5 implicit conversions from `void *` (valid in C, not
in C++).  Not vendored: kernels that the
available CPUs cannot execute (all `knl` kernels use AVX-512PF prefetches, `bulldozer` and
`piledriver` use FMA4) and kernels of the `zen`, `zen2`, `zen3`
configurations, which use the haswell kernels.

## Files

| File | Contents |
|---|---|
| `select.h` | the table of kernels (MR, NR, block sizes, alignment, interface); sets `USE_BLIS_DGEMM` |
| `select_s.h`, `select_c.h`, `select_z.h` | the same for SGEMM, CGEMM, ZGEMM (`USE_BLIS_SGEMM`, …) |
| `ugemm.h` | the adapter: ulmBLAS micro kernel interface for the selected kernel |
| `xugemm.h` | the adapter for the types other than double |
| `compat.h`, `compat_end.h` | the definitions of `blis.h` that the kernels use (types, `auxinfo_t`, the edge case macros `GEMM_UKR_SETUP_CT*`), written for ulmBLAS; undefined again after each kernel |
| `kernel/NAME.h` | the vendored kernels, generated by `mk/vendor_blis_kernels.py` |
| `asm/bli_x86_asm_macros_*.h` | BLIS's assembly macro headers (BLIS 2.0 and AOCL-BLAS 5.0) |

### Provenance

`mk/vendor_blis_kernels.py BLIS_DIR AOCL_DIR AOCL51_DIR` generates `kernel/` and `asm/`
from unpacked upstream trees, which it only reads:

| Tree | Repository | Tag | Commit |
|---|---|---|---|
| BLIS 2.0 | https://github.com/flame/blis | `2.0` | e8566eb3e773fb54d11b33e371d13f22d2941e50 |
| AOCL-BLAS 5.0 | https://github.com/amd/blis | `5.0` | 34d4bbade33a4384cfaff2208c833fc33a311c5d |
| AOCL-BLAS 5.1 (DGEMM: `zen5_8x24` only; all AOCL kernels for the other types) | https://github.com/amd/blis | `5.1` | 16f852a065e76e824d77bc39e2baa82ac19ed419 |

Each generated file starts with the unchanged license comment of its source
file, followed by the repository, tag, commit, path and SHA-256 of the
source file and the list of the mechanical changes: `#include` lines
removed (`bli_x86_asm_macros.h` replaced in place by the copy in `asm/`),
the functions for other data types removed, the code wrapped in namespace
`ulmBLAS::blis::kernel::NAME` with the kernel declared `inline`, and all
macros undefined at the end.  The code of the kernel function itself is
not changed.  In the copies of the assembly macro headers only the
lowercase aliases `and()`, `or()`, `xor()` are excluded for C++ (operator
names), which no kernel uses.

### License

The kernels and the assembly macro headers are distributed under the BSD
3-Clause License of BLIS and AOCL-BLAS, with the copyright notices of
The University of Texas at Austin and Advanced Micro Devices, Inc. in each
file; see section 7 of [`LICENSE`](../../../../LICENSE).

## The adapter

The ulmBLAS micro kernel computes C ← beta·C + alpha·A·B for one full MR x NR
tile from an MR x kc panel of A and a kc x NR panel of B, packed as BLIS
packs them (k-major panels, zero padded); partial tiles are handled by the
buffered variant in `../ugemm.tcc`.  `ugemm.h`

- calls kernels with the interface of BLIS 1.0 and newer (`m, n, k`, void
  pointers) with m = MR, n = NR; their edge case macros use a local tile if
  the storage of C is not supported by the assembly code;
- calls the AOCL kernels (interface of BLIS 0.x, full tiles only) directly
  and uses a local tile if C has neither unit row nor unit column stride
  and the kernel does not support general strides;
- passes `nextA` and `nextB` as BLIS's `auxinfo_t` (next panel of A and B,
  used for prefetching);
- computes alpha·A·B into a local tile and updates C in C++ if beta is NaN.
  The kernels branch on beta == 0 with `vcomisd`/`je`, which is also taken
  for NaN, and would then not read C; ulmBLAS's own kernels propagate NaN.

The packing buffers are aligned to 64 bytes for the AVX-512 kernels
(`../../../auxiliary/memorypool.tcc`).  With a BLIS kernel the SSE TRSM
micro kernels (which require MR = NR = 4 for double) are replaced by the
reference TRSM micro kernels.

## Tests

Every kernel has two test configurations in `test/config/`: `blis-NAME`
(default block sizes) and `blis-NAME-small` (small MC, KC = 11, small NC:
many partial blocks, panels and k remainders).  They are not part of the
default configurations; run them with, e.g.,

```sh
make check CONFIGS="blis-skx_16x14 blis-skx_16x14-small"
```

The configurations require the CPU features of the kernel and are reported
as SKIP on other CPUs.

## ZGEMM kernels

ZGEMM (`std::complex<double>`) kernels are selected independently of the
DGEMM kernel, with exactly one `-DULMBLAS_ZGEMM_UKERNEL_<origin>_<name>`
(`select_z.h`); both selections can be combined.  Without one, ZGEMM uses
the reference micro kernel as before.

```sh
make cblas ARCHFLAGS="-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_haswell_3x4"
make cblas ARCHFLAGS="-mavx -mfma -mavx512f -mavx512dq -DULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4"
```

| Name (`ULMBLAS_ZGEMM_UKERNEL_…`) | Origin | Upstream function | ISA | MR x NR | Default MC/KC/NC (configuration) | Storage of C in assembly |
|---|---|---|---|---|---|---|
| `blis_haswell_3x4` | BLIS 2.0 | `bli_zgemm_haswell_asm_3x4` | AVX, FMA | 3 x 4 | 192/256/4080 (haswell) | row major (row preference) |
| `blis_haswell_4x3` | BLIS 2.0 | `bli_zgemm_haswell_asm_4x3` | AVX, FMA | 4 x 3 | 36/256/4080 (haswell 8x6 family: MC 44) | column major |
| `blis_sandybridge_4x4` | BLIS 2.0 | `bli_zgemm_sandybridge_asm_4x4` | AVX | 4 x 4 | 64/192/4096 (sandybridge) | column major |
| `blis_aocl_haswell_3x4` | AOCL-BLAS 5.1 | `bli_zgemm_haswell_asm_3x4` (AMD's version, BLIS 0.x interface) | AVX, FMA | 3 x 4 | 12/566/256 (zen3: MC 18) | row major, general (row preference) |
| `blis_zen4_12x4` | AOCL-BLAS 5.1 | `bli_zgemm_zen4_asm_12x4` | AVX-512F, AVX-512DQ | 12 x 4 | 60/512/2004 (zen4, zen5) | column, row major, general |
| `blis_zen4_4x12` | AOCL-BLAS 5.1 | `bli_zgemm_zen4_asm_4x12` | AVX-512F, AVX-512DQ | 4 x 12 | 36/512/2004 (zen4 TRSM: MC 40) | row major, general (row preference) |
| `openblas_haswell_4x2` | OpenBLAS 0.3.34 | `zgemm_kernel_4x2_haswell.c` | AVX2, FMA | 4 x 2 (4 x 6 tiles inside) | 192/192/4096 (HASWELL P, Q) | column major |
| `openblas_skylakex_4x2` | OpenBLAS 0.3.34 | `zgemm_kernel_4x2_skylakex.c` | AVX-512F | 4 x 2 (4 x 12 tiles inside) | 256/128/4096 (SKYLAKEX P, Q) | column major |

MR x NR in complex elements.  `-DULMBLAS_ZGEMM_{MC,KC,NC}=…` override the
block sizes.  OpenBLAS computes its R (NC) at run time from its buffer
size (more than 40000 for ZGEMM); NC = 4096 blocks every n <= 4096 as
OpenBLAS does and bounds the packing buffer.

**Packing and conjugation.**  The kernels read panels of interleaved
(re, im) elements, k-major, which is ulmBLAS's packing of
`std::complex<double>` (BLIS: `dcomplex`, the same layout); ulmBLAS
conjugates A and B while packing, so every kernel runs as its variant
without conjugation (OpenBLAS: compiled with `NN`; the BLIS kernels have no
conjugation parameter).  With NR = 2 ulmBLAS packs B exactly as OpenBLAS's
`zgemm_ncopy_2` does (consecutive panels of 2 columns), so the OpenBLAS
kernels are used without any change of their code; the macro kernel calls
them once per packed block (`xugemm_block`, as for `openblas_haswell_4x8`;
`-DULMBLAS_ZGEMM_NO_PANEL` switches this off).

**Row preference.**  `bli_zgemm_haswell_asm_3x4` stores row-major tiles
only (`GEMM_UKR_SETUP_CT`); for column-major C every tile would go through
a local tile.  As BLIS does, ZGEMM computes C^T = B^T A^T instead when the
selected kernel prefers rows (`ULMBLAS_BLIS_ZGEMM_ROWPREF`, `gemm.tcc`):
an exchange of arguments, no copy.  The other level 3 routines keep C as
it is (the kernels handle every storage, through a local tile if needed).

**Vendoring.**  `mk/vendor_blis_kernels.py` and
`mk/vendor_openblas_kernels.py` generate `kernel/z_*.h` and
`../openblas/kernel/z_*.h` from the unmodified upstream trees; the
changes are listed in each file.  Besides those of the DGEMM kernels:
the BLIS complex kernels test beta = 0 with `and(r8b, r9b)`, the lowercase
alias of the assembly macro `AND()`, which is an operator name in C++ and
cannot be a macro; the script replaces it by `AND(`, which emits the same
instruction.  `compat.h` provides `scomplex`, `dcomplex`, the edge case
macros for every type character and AOCL's `mulfactor` and (empty) trace
macros.

**Adapter.**  `xugemm.h` is the adapter for the types other than double
(traits `Kernel<T>`, no run-time dispatch); it supports the same three
interfaces as the DGEMM adapter in `ugemm.h`, which is unchanged for
double.  As there, alpha*A*B is computed into a local tile if beta is NaN
(the complex kernels test beta == 0 with `vucomisd`/`sete`, which also
holds for NaN).

**Tests.**  Every kernel has the test configurations `z-NAME` and
`z-NAME-small` (KC = 11, small MC and NC); in the latter the selected
kernel fixes MR and NR for `std::complex<double>`.  T0354 (micro kernel,
alpha and beta -1, 0, 1, real, imaginary, complex) and T0355 (GEMM with
all tile and k remainders, all transpositions including conjugate
transposition) were added for these kernels.

## SGEMM kernels

SGEMM (`float`) kernels are selected with exactly one
`-DULMBLAS_SGEMM_UKERNEL_<origin>_<name>` (`select_s.h`), independently of
the DGEMM, CGEMM and ZGEMM selections.  They use the adapter `xugemm.h`,
and the OpenBLAS kernels are called once per packed block, as for ZGEMM;
the test configurations are `s-NAME` and `s-NAME-small`.

```sh
make cblas ARCHFLAGS="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_6x16"
make cblas ARCHFLAGS="-mavx -mfma -mavx512f -DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4"
```

| Name (`ULMBLAS_SGEMM_UKERNEL_…`) | Origin | Upstream function | ISA | MR x NR | Default MC/KC/NC (configuration) | Storage of C in assembly |
|---|---|---|---|---|---|---|
| `blis_haswell_6x16` | BLIS 2.0 | `bli_sgemm_haswell_asm_6x16` | AVX, FMA | 6 x 16 | 144/256/4080 (haswell: MC 168) | row or column major (row preference) |
| `blis_haswell_16x6` | BLIS 2.0 | `bli_sgemm_haswell_asm_16x6` | AVX, FMA | 16 x 6 | 96/256/4080 (haswell 8x6 family: MC 112) | column major |
| `blis_haswell_24x4` | BLIS 2.0, `kernels/haswell/3/old/` | `bli_sgemm_haswell_asm_24x4` | AVX, FMA | 24 x 4 | 168/256/4080 (no BLIS configuration uses it: haswell values) | any |
| `blis_haswell_4x24` | BLIS 2.0, `kernels/haswell/3/old/` | `bli_sgemm_haswell_asm_4x24` | AVX, FMA | 4 x 24 | 168/256/4080 (as `haswell_24x4`) | any (row preference) |
| `blis_sandybridge_8x8` | BLIS 2.0 | `bli_sgemm_sandybridge_asm_8x8` | AVX | 8 x 8 | 128/384/4096 (sandybridge) | column major |
| `blis_penryn_8x4` | BLIS 2.0 | `bli_sgemm_penryn_asm_8x4` | SSE3 | 8 x 4 | 768/384/4096 (penryn) | column major, 16-byte aligned |
| `blis_skx_32x12_l2` | BLIS 2.0 | `bli_sgemm_skx_asm_32x12_l2` | AVX-512F | 32 x 12 | 480/384/3072 (skx) | column major |
| `blis_aocl_haswell_6x16` | AOCL-BLAS 5.1 | `bli_sgemm_haswell_asm_6x16` (AMD's version, BLIS 0.x interface) | AVX, FMA | 6 x 16 | 144/256/4080 (zen3) | any (row preference) |
| `openblas_haswell_8x4` | OpenBLAS 0.3.34 | `sgemm_kernel_8x4_haswell_2.c` | AVX2, FMA | 8 x 4 (block calls) | 320/320/4096 (HASWELL P, Q) | column major |
| `openblas_skylakex_16x4` | OpenBLAS 0.3.34 | `sgemm_kernel_16x4_skylakex_3.c` | AVX-512F | 16 x 4 (block calls) | 448/448/4096 (SKYLAKEX P, Q) | column major |

MC is rounded down to a multiple of MR and NR where the upstream value is
not one.  The old kernels 24 x 4 and 4 x 24 get local labels as the
DGEMM kernels of `old/`; the OpenBLAS kernels are used without any change
of their code (`../openblas/kernel/s_*.h`).  AOCL-BLAS's zen4 and zen5
configurations use its copy of `bli_sgemm_skx_asm_32x12_l2`, which is not
vendored separately.

## CGEMM kernels

CGEMM (`std::complex<float>`) kernels are selected the same way, with
exactly one `-DULMBLAS_CGEMM_UKERNEL_<origin>_<name>` (`select_c.h`),
independently of the DGEMM, SGEMM and ZGEMM selections.  Packing,
conjugation, row preference, block calls, vendoring (including the `and(`
replacement), adapter and tests are as for ZGEMM above, with `scomplex`
for `dcomplex`; the test configurations are `c-NAME` and `c-NAME-small`.

```sh
make cblas ARCHFLAGS="-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8"
make cblas ARCHFLAGS="-mavx -mfma -mavx512f -DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2"
```

| Name (`ULMBLAS_CGEMM_UKERNEL_…`) | Origin | Upstream function | ISA | MR x NR | Default MC/KC/NC (configuration) | Storage of C in assembly |
|---|---|---|---|---|---|---|
| `blis_haswell_3x8` | BLIS 2.0 | `bli_cgemm_haswell_asm_3x8` | AVX, FMA | 3 x 8 | 72/256/4080 (haswell: MC 75) | row major (row preference) |
| `blis_haswell_8x3` | BLIS 2.0 | `bli_cgemm_haswell_asm_8x3` | AVX, FMA | 8 x 3 | 48/256/4080 (haswell 8x6 family: MC 56) | column major |
| `blis_sandybridge_8x4` | BLIS 2.0 | `bli_cgemm_sandybridge_asm_8x4` | AVX | 8 x 4 | 96/256/4096 (sandybridge) | column major |
| `blis_aocl_haswell_3x8` | AOCL-BLAS 5.1 | `bli_cgemm_haswell_asm_3x8` (AMD's version, BLIS 0.x interface) | AVX, FMA | 3 x 8 | 144/256/4080 (zen3, zen4) | row major, general (row preference) |
| `openblas_haswell_8x2` | OpenBLAS 0.3.34 | `cgemm_kernel_8x2_haswell.c` | AVX2, FMA | 8 x 2 (block calls) | 256/256/4096 (HASWELL P, Q) | column major |
| `openblas_skylakex_8x2` | OpenBLAS 0.3.34 | `cgemm_kernel_8x2_skylakex.c` | AVX-512F | 8 x 2 (block calls) | 384/192/4096 (SKYLAKEX P, Q) | column major |

MR x NR in complex elements; MC is rounded down to a multiple of MR and
NR where the upstream value is not one.  BLIS's skx configuration and
AOCL's zen4 and zen5 configurations register no other complex float
kernel (BLIS skx computes complex products with its 1m method, AOCL zen4
registers `bli_cgemm_haswell_asm_3x8`).
