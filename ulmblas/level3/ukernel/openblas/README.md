# OpenBLAS-derived GEMM micro kernels

This directory contains GEMM kernels of
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS): two DGEMM kernels,
described here first, and kernels for float, complex float and complex
double ([below](#sgemm-cgemm-and-zgemm-kernels)).  The DGEMM kernels are
adapted so that
ulmBLAS can call them as micro kernels inside its own GEMM implementation
(packing, blocking, macro kernel of ulmBLAS): the kernel for Intel
Skylake-X and newer AVX-512 CPUs (target `SKYLAKEX`, also used for
`COOPERLAKE`) and the kernel for AVX2/FMA CPUs (target `HASWELL`, which
OpenBLAS also uses for Broadwell, Skylake without AVX-512, Zen 1 to 3 and
Alder Lake).  They are **not original ulmBLAS kernels** and not BLIS
kernels; they are selected through the same mechanism as the BLIS-derived
kernels ([`../blis`](../blis/README.md)):

```sh
make cblas ARCHFLAGS="-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12"
make cblas ARCHFLAGS="-mavx -mavx2 -mfma -DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8"
```

| Name | Upstream | ISA | MR x NR | Default MC/KC/NC | Status |
|---|---|---|---|---|---|
| `skylakex_16x12` | `kernel/x86_64/dgemm_kernel_16x2_skylakex.c` of OpenBLAS 0.3.34 | AVX-512F | 16 x 12 | 192/384/8640 (OpenBLAS SKYLAKEX: `DGEMM_DEFAULT_P/Q/R`) | verified: passes the test suite on an Intel Xeon Platinum 8358 (Ice Lake) and an AMD EPYC 9454 (Zen 4), GCC 14.2 (no sanitizer run yet) |
| `haswell_4x8` | `kernel/x86_64/dgemm_kernel_4x8_haswell.S` of OpenBLAS 0.3.34 (assembly file, assembled unchanged) | AVX2, FMA | 4 x 8 (4 x 12 inside the kernel, see below) | 512/256/13824 (OpenBLAS HASWELL: `DGEMM_DEFAULT_P/Q/R`) | verified: passes the test suite (configurations `openblas-haswell_4x8`, `-small`, `-kc600`) on an Intel Core i5-12500 (Alder Lake, AVX2), GCC 12.2, also with ASan/UBSan, and the numeric tests under valgrind memcheck without errors (valgrind also checks the loads and stores of the assembly code); the assembled machine code is identical to the `dgemm_kernel` of native OpenBLAS 0.3.34 (HASWELL) |

The rest of this section describes `skylakex_16x12`; `haswell_4x8` is
described [below](#the-haswell-kernel-haswell_4x8).

## The upstream kernel

In OpenBLAS 0.3.34 the target `SKYLAKEX` builds the DGEMM kernel from
`dgemm_kernel_16x2_skylakex.c` (inline assembly; `KERNEL.SKYLAKEX`), with A
packed by `gemm_ncopy_16` (panels of 16 rows, k-major, as ulmBLAS packs
them for MR = 16) and B by `gemm_ncopy_2` (panels of **2** columns).  The
"16x2" in the name refers to the packing; the function
`CNAME(m, n, k, alpha, A, B, C, ldc)` is a macro kernel: it computes
C ← C + alpha·A·B for a whole packed block, in tiles of **16 x 12**
(remainders 8, 4, 2, 1 rows and 10, 8, 6, 4, 2, 1 columns).  beta is
applied by OpenBLAS before (`dgemm_beta_skylakex.c`).

The 16 x 12 tile uses 24 zmm accumulators (zmm8-zmm31).  Unless
`BROADCAST_KERNEL` is defined (it is not, anywhere in OpenBLAS 0.3.34), the
kernel does not broadcast single elements of B: per k it loads A with four
`vmovddup` (pairs of duplicated elements of the 16 rows) and the 12
elements of B with six `vbroadcastf32x4` (one pair of columns per 128-bit
lane) and computes 24 `vfmadd231pd`, i.e. 10 loads per 24 FMAs (BLIS skx
16 x 14: 2 loads and 14 broadcasts per 28 FMAs).  The store de-interleaves
the accumulators with `vunpcklpd`/`vunpckhpd`.  It prefetches A 384 and 448
bytes ahead, the tile of C into L2 during the main loop over k
(`prefetcht1`, with a changing stride) and into L1 during the last 16
iterations (`prefetcht0`), and the next block of B into L2.

## Changes for ulmBLAS

`mk/vendor_openblas_kernels.py OPENBLAS_DIR` generates `kernel/NAME.h` from
an unmodified OpenBLAS tree (tag v0.3.34, commit
e0166008be8e466242aa76b2ff75ce3f0fbf574a; the generated file records the
SHA-256 of the source file).  Changes:

- **B addressing** (the only change of the assembly code, 3 exact text
  replacements): the kernel reads the 12 columns of B as 6 panels of 2
  columns, 16·K bytes apart (through the pointers `%1`, `%1+r12`,
  `%1+2·r12`, `r15 = %1+3·r12`, `r15+r12`, `r15+2·r12`) and advances `%1`
  and `r15` by 16 bytes per k.  ulmBLAS packs a panel of NR = 12 columns
  k-major (12 consecutive elements per k).  With `r12` = 16 bytes and
  advances of 96 bytes per k the same pointers address the 6 column pairs
  of a row of 12 elements.  The computational instructions, the register
  allocation, the loop structure and the prefetching are unchanged.  After
  the change only the 16 x 12 path addresses B correctly; ulmBLAS calls the
  kernel with full 16 x 12 tiles only.
- `#include "common.h"` removed (`compat.h` defines `BLASLONG`), the code
  is wrapped into a namespace, the kernel function is declared `inline`,
  and all macros of the file are undefined at the end.

Packing B in OpenBLAS's layout instead would also change the packed panels
that ulmBLAS's TRSM routines address by row offsets and update in place.

The adapter (`../blis/ugemm.h`, interface 2) calls the kernel with
m = 16, n = 12, k = kc for column-major C (ldc = the column stride); it
scales the tile of C by beta before (sets it to zero for beta = 0, without
reading C) and uses a local tile for other storage of C.

In addition, ulmBLAS's macro kernel `mgemm` calls the kernel once per
panel of B for all full 16-row tiles of the block (m = 16·tiles, n = 12;
`blisk::ugemm_panel`), as OpenBLAS's driver calls it for a block; partial
tiles and the other level 3 routines use the per-tile adapter.  Calling it
once per tile costs 2-3%
([016-2026-10-27-uc3-icelake-openblas-kernel-prof](../../../../benchmark_results/016-2026-10-27-uc3-icelake-openblas-kernel-prof/README.md)).
`-DULMBLAS_DGEMM_NO_PANEL` switches the panel calls off.  DGEMM
(`ulmblas/level3/gemm.tcc`) also packs B interleaved with the computation
for the first block of A, as OpenBLAS does: each panel of 12 columns is
packed and used right away; `-DULMBLAS_DGEMM_NO_INTERLEAVE` packs the
whole block of B first.  With both, ulmBLAS reaches 0.970 of native
OpenBLAS
([021-2026-10-27-uc3-icelake-openblas-kernel-interleave](../../../../benchmark_results/021-2026-10-27-uc3-icelake-openblas-kernel-interleave/README.md)).  With the panel
calls ulmBLAS reaches 0.962 of native OpenBLAS on an Intel Xeon Platinum
8358 for n ≥ 1000, with one call per tile 0.940
([020-2026-10-27-uc3-icelake-openblas-kernel-panel](../../../../benchmark_results/020-2026-10-27-uc3-icelake-openblas-kernel-panel/README.md)).

## The Haswell kernel (`haswell_4x8`)

In OpenBLAS 0.3.34 the target `HASWELL` builds the DGEMM kernel from the
assembly file `dgemm_kernel_4x8_haswell.S` (`KERNEL.HASWELL`), with
`DGEMM_UNROLL_M = 4`, `DGEMM_UNROLL_N = 8`: A packed in panels of 4 rows
(`gemm_ncopy_4`), B in panels of 8 columns (`dgemm_ncopy_8_skylakex.c`),
both k-major, i.e. exactly the layout ulmBLAS's packing produces for MR =
4, NR = 8.  Like the SKYLAKEX kernel, the function
`CNAME(m, n, k, alpha, A, B, C, ldc)` is a macro kernel for a whole packed
block, C ← C + alpha·A·B, column-major C (beta is applied before).
Inside, it takes the columns of B in groups of **24**: for each group it
copies 12 columns (the 8 columns of one panel and 4 of the next) into a
buffer on its stack (12 consecutive elements per k), computes **4 x 12**
tiles from it (12 ymm accumulators; per k 3 loads of B, 4 broadcasts of
A, 12 FMAs), then the next 12 columns.  Remaining columns are computed in
tiles of 4 x 8, 4 x 4, 4 x 2 and 4 x 1, remaining rows with 2 and 1 rows.

**Integration.**  The file is assembled as it is
(`mk/vendor_openblas_kernels.py` generates `asm/haswell_4x8.S` from the
unmodified OpenBLAS tree and records the SHA-256 of the source): only
`#include "common.h"` is replaced by `asm/compat.h`, which defines the
macros of OpenBLAS's common headers the file uses (function prologue and
epilogue for ELF and Mach-O, `ALIGN_4`, `ALIGN_5`, `SIZE`, `BASE_SHIFT`)
as OpenBLAS defines them for double precision on x86-64, and the
function name `ulmblas_openblas_dgemm_kernel_haswell_4x8` (no clash with
OpenBLAS's own symbols when both libraries are linked).  `TRMMKERNEL` is
not defined, so only the GEMM kernel is assembled.  The calling
convention is OpenBLAS's: System V x86-64 (m, n, k in `rdi`, `rsi`,
`rdx`, alpha in `xmm0`, A, B, C in `rcx`, `r8`, `r9`, ldc on the stack);
the kernel saves the callee-saved registers it uses (`rbx`, `rbp`,
`r12`-`r15`); `kernel/haswell_4x8.h` declares it.

**Build.**  If the selection macro appears in the flags (`ARCHFLAGS`,
`CPPFLAGS`, `CXXFLAGS` or a test configuration), `make.inc` adds the
assembled object to `libcblas.a`, `libclapack.a` and `libf77blas.a`
(`ULMBLAS_ASM_OBJECTS`); other builds are unchanged.  A program that
uses the C++ templates of ulmBLAS directly with this kernel and does not
link one of these libraries must assemble `asm/haswell_4x8.S` itself
(`c++ -I<ulmBLAS> -c haswell_4x8.S`).

**Limit on k.**  The stack buffer holds 12·k doubles for k ≤ 298
(`L_BUFFER_SIZE` = 256·8·12 + 4096 bytes; OpenBLAS calls the kernel with k
≤ Q = 256).  The adapter (`../blis/ugemm.h`, `ULMBLAS_BLIS_DGEMM_KMAX` =
256) splits longer k loops of a tile into several calls (the panels are
k-major, so a part of the k loop starts at A + l·MR, B + l·NR), and the
panel calls of the macro kernel fall back to tiles; GEMM itself uses KC =
256.

**Calls.**  The adapter calls the kernel per tile with m = 4, n = 8
(partial tiles, the other level 3 routines).  The macro kernel `mgemm`
calls it **once per packed block** for all full tiles (m = 4·tiles, n =
8·panels; `blisk::ugemm_block`, `ULMBLAS_BLIS_DGEMM_BLOCKCALL`), as
OpenBLAS's driver does, and DGEMM packs B interleaved with the first
block of A in groups of 3 panels (24 columns,
`ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS`), as OpenBLAS's driver.  The
kernel computes 4 x 12 tiles only for groups of 24 columns in one call;
called once per panel of B (n = 8, the mechanism of `skylakex_16x12`) it
uses its 4 x 8 path for every tile, which made ulmBLAS 13% slower than
native OpenBLAS on an Intel Core i5-12500 (0.86-0.87 for n ≥ 256; with
the block calls 0.98-1.00;
[024-2026-10-27-heim-openblas-haswell-kernel](../../../../benchmark_results/024-2026-10-27-heim-openblas-haswell-kernel/README.md),
which also compares the loop nesting of both drivers).  `-DULMBLAS_DGEMM_NO_PANEL` switches the
block calls off (one call per tile).

## SGEMM, CGEMM and ZGEMM kernels

The kernels for the other types are generated by the same script from the
same OpenBLAS tree (tag v0.3.34); each file records the source path and
its SHA-256 and lists the changes:

| File | Upstream (`kernel/x86_64/`) | Selection |
|---|---|---|
| `kernel/s_haswell_8x4.h` | `sgemm_kernel_8x4_haswell_2.c` | `-DULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4` |
| `kernel/s_skylakex_16x4.h` | `sgemm_kernel_16x4_skylakex_3.c` | `-DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4` |
| `kernel/c_haswell_8x2.h` | `cgemm_kernel_8x2_haswell.c` | `-DULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2` |
| `kernel/c_skylakex_8x2.h` | `cgemm_kernel_8x2_skylakex.c` | `-DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2` |
| `kernel/z_haswell_4x2.h` | `zgemm_kernel_4x2_haswell.c` | `-DULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2` |
| `kernel/z_skylakex_4x2.h` | `zgemm_kernel_4x2_skylakex.c` | `-DULMBLAS_ZGEMM_UKERNEL_openblas_skylakex_4x2` |

Unlike `skylakex_16x12`, their code is not changed: with NR equal to
OpenBLAS's unroll factor for n, ulmBLAS packs B as OpenBLAS does.  The
changes are those of the wrapping (includes, namespace, `inline`, macros
undefined at the end); the complex kernels are compiled with `NN`
defined, their variant without conjugation, since ulmBLAS conjugates A
and B while packing.  The macro kernel calls them once per packed block.
Block sizes, the adapter (`../blis/xugemm.h`) and the tests are described
with the other kernels of each type in
[`../blis/README.md`](../blis/README.md#zgemm-kernels) (ZGEMM,
[SGEMM](../blis/README.md#sgemm-kernels),
[CGEMM](../blis/README.md#cgemm-kernels)).

## License

OpenBLAS is distributed under the BSD 3-Clause License (Copyright (c)
2011-2014, The OpenBLAS Project).  The SKYLAKEX kernel file has no license
comment of its own, so the generated file starts with the license of the
repository, as do the files for the other types; `asm/haswell_4x8.S`
keeps the license comment of its source (Copyright (c) 2015, The OpenBLAS
Project, BSD 3-Clause).  See section 8 of [`LICENSE`](../../../../LICENSE).
