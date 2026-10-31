# BLAS and CBLAS conformance audit

Audit of all public entry points of the BLAS interfaces of ulmBLAS (148
routines; Fortran 77 interface, CBLAS names and `ulmblas_` names of the C
interface) against their reference implementations, October 2026.  The
findings are registered in [`known_issues.md`](known_issues.md) (#0016 to
#0029); this file records the method, the conventions that were
established, the classification of the differences and a checklist of what
was checked for every routine.

## Method

- **References.**  Netlib reference BLAS and CBLAS of LAPACK 3.12.1
  (`BLAS/SRC`, `CBLAS/src`, `CBLAS/include`); the development version of
  Reference-LAPACK (master 472b3a0) for `i?amax` with NaN (#0014) and
  `?rotmg` (#0011), which ulmBLAS follows deliberately.
- **Source reading.**  For every entry point the reference routine and, for
  the CBLAS names, the reference wrapper (`cblas_*.c` and the `*sub.f`
  helpers) were read and compared with the ulmBLAS entry point and the C++
  templates and kernels it calls: argument checks and the reported
  parameter positions, quick returns, alpha/beta special cases, pointer
  adjustment and traversal for positive, zero and negative increments,
  leading dimensions, storage formats (full, band, packed, triangles,
  unit diagonals, Hermitian diagonals), operation flags and the mapping
  of row major calls, return values and index conventions.
- **Differential tests** (`make check-conformance`, see
  [conformance/](conformance)).  The same calls against ulmBLAS and
  against the reference BLAS/CBLAS built from the LAPACK 3.12.1 sources;
  every operand lies in a buffer with guard cells, the unreferenced parts
  of matrices are NaN, and all buffers are compared after each call
  (bitwise up to a relative tolerance for rounding).  Level 1: all
  routines with n in {0, 1, 3} and increments -2..2 (4440 calls).  Level 2
  and 3: all routines, all flags, both storage orders and the Fortran 77
  interface, dimensions {0, 1, 5} (level 3: {0, 1, 5, 9}), several leading
  dimensions, alpha/beta in {0, 1, other} with NaN in the operands that
  must not be read (about 217000 calls).
- **Declarations.**  `make check-conformance` declares every prototype of
  the reference cblas.h again after the cblas.h of ulmBLAS (a different
  type is a compile error); test T0503 compiles a C client in the style of
  the standard with `-Wall -Werror`, T0904 a C++ client with
  `std::complex`.
- **Reference test programs.**  The BLAS and CBLAS test programs of LAPACK
  3.12.1 (`make check-reference-lapack`) and the level 1 test programs of
  the development version (for `i?amax`).
- **Regression tests** for every confirmed defect (they fail with the code
  before the fix), DESIGN tests for open questions, and the numerical
  tests with AddressSanitizer and UndefinedBehaviorSanitizer.

## Conventions of the interfaces

| | Fortran 77 | CBLAS names | `ulmblas_` names |
|---|---|---|---|
| contract | reference BLAS | reference CBLAS | ulmBLAS (column major, no argument checks, preconditions in cblas.h, #0029) |
| incx < 0, level 1 | start index 1-(n-1)*incx; `?asum`, `?scal`, `i?amax`: return 0 / nothing | as Fortran 77 | BLAS storage order for all functions |
| incx = 0, level 1 | `?asum`, `i?amax`: 0; `?scal`: nothing; others: x(1) repeated | as Fortran 77 | x[0] repeated |
| incx = 0, level 2 | error | error | not checked |
| `i?amax` | 1-based, 0 for n < 1 or incx <= 0 | 0-based, 0 for n <= 0 or incx <= 0 | 0-based, -1 for n <= 0 |
| invalid arguments | XERBLA prints and returns (reference: stops) | cblas_xerbla prints and returns (reference: exits) | none |
| row major | – | as the reference wrappers (positions as for the column major call, RowMajorStrg) | – |

The internal C++ templates support general strides (also negative and
zero increments and strides of matrices); the restrictions above are
applied in the public interfaces only.

## Findings

Defects, fixed on this branch (each with a regression test that fails
without the fix):

| Issue | Routines | Defect |
|---|---|---|
| #0016 | ?syr2k, ?her2k (all interfaces) | B packed with the leading dimension of A: wrong results, reads outside B |
| #0017 | ?asum, ?scal, csscal, zdscal, i?amax, cblas_?nrm2 | increments <= 0: reads and writes outside x, results differ from the reference |
| #0018 | ?rotg | overflow for large operands, wrong values for Inf (algorithm before LAPACK 3.10) |
| #0019 | cblas_?syr, ?syr2, ?her, ?her2 | ldA = 0 accepted for n = 0 |
| #0020 | CBLAS level 3, cblas_xerbla | row major errors: RowMajorStrg not set, positions not translated |
| #0021 | cblas_?her2k | row major: wrote to alpha |
| #0022 | cblas.h | non-const input arguments of ?symm, ?hemm, ?her2k |

Decisions on the design questions (see the registry for details):

| Issue | Decision |
|---|---|
| #0024 | reference semantics routine by routine; documented quick returns kept, no data-dependent skipping of zero elements: the skips that only ulmBLAS made (gemv, gbmv, hemv, ger, spr, hpr, tbsv, tpsv, row major hpr2) are removed; where the reference skips (e.g. DTRMV, DSYR, DSYRK, DTRMM), ulmBLAS computes (IEEE propagation of NaN and Inf) |
| #0025 | `AtlasConj` and 'R' are documented extensions; all other invalid codes are rejected |
| #0026 | `?scal` multiplies also for alpha = 0, as the reference |
| #0027 | trsm keeps the reciprocals of the diagonal (division: slightly more accurate, dtrsm up to 2.5% and ztrsm up to 49% slower on the Apple M5); to be reconsidered with x86-64 measurements |
| #0028 | cblas.h source compatible with the CBLAS standard without ABI change: `void *` for complex arguments of the CBLAS names, the standard type names, `CBLAS_INDEX` = the index type of the build; all prototypes are compared with the reference cblas.h by `make check-conformance` |
| #0029 | no argument checks in the `ulmblas_` functions; preconditions documented in cblas.h |

Open: #0023 (trmm: 0*Inf from the zeros of packed diagonal blocks; XFAIL
test).  The registry compares ulmBLAS with the reference BLAS, oneMKL,
AOCL-BLAS, Accelerate, BLIS and OpenBLAS (`conformance/trmm_inf.c`): only
the reference and oneMKL avoid the NaN; the question whether LAPACK may
rely on it is open.  The alternatives for a fix are listed there as well.

Differences that are accepted as implementation-specific (not registered):

- Rounding: other order of summation and evaluation (blocked kernels, fused
  operations such as `alpha*(sum)` in `axpyf`, `(s+|re|)+|im|` in
  `dzasum`), fused multiply-adds (gfortran contracts by default, the C++
  compilers within expressions); in rare cases this changes overflow or
  underflow of intermediate results.
- Complex arithmetic with Inf: `std::complex` follows C Annex G
  (e.g. (Inf,0)*(1,0) = (Inf,NaN)), gfortran uses the formulas without
  special cases and converts a real divisor to complex ((Inf, x)/2 has a
  NaN component); results with infinite complex operands can differ.
- Signs of zeros: the reference CBLAS conjugates vectors in place for
  complex row major calls with ConjTrans (gemv, gbmv, hemv, hbmv, hpmv,
  tr*mv, tr*sv), so zero imaginary parts can come back as -0.

Behavior of the reference CBLAS that ulmBLAS does not reproduce:

- `cblas_cgemv`, `cblas_zgemv`, `cblas_cgbmv`, `cblas_zgbmv` with row major
  storage, ConjTrans, M = 0 and N > 0 negate y with loop bounds that are
  not initialized (`cblas_cgemv.c`, the y loop after the Fortran call; the
  loop before it is guarded by M > 0): undefined behavior, a hang in the
  differential test.  ulmBLAS returns.
- The temporary conjugated copies of the row major wrappers have unit
  increment, so incX = 0 (gemv, gbmv, hemv, hbmv, hpmv, her, hpr) or
  incY = 0 (gerc, her2, hpr2) is not reported; ulmBLAS reports it.
- Row major `cblas_?gerc` reports errors as `cblas_?geru`; row major
  complex ?syrk/?syr2k accept ConjTrans and ?herk/?her2k accept Trans;
  an invalid uplo of row major ?syrk, ?syr2k, ?herk is reported as
  parameter 3.  ulmBLAS reports the routine called, rejects these values
  (as the column major reference) and reports uplo as parameter 2.

## Checklist

Legend: **R** source reading against the reference routine (Fortran 77)
or wrapper (CBLAS, both storage orders); **D** differential test against
the reference (Fortran 77, CBLAS column and row major); **N** BLAS/CBLAS
test programs of LAPACK 3.12.1; **U** `ulmblas_` names read (no reference
contract); numbers: issues found.  All four types (s, d, c, z) of a row
were checked unless the row lists fewer.

### Level 1

| Routine | Types | F77 | CBLAS | ulmblas_ | Issues |
|---|---|---|---|---|---|
| rotg | s, d, c, z | R D N | R D N | U | #0018 |
| rotmg | s, d | R N | R N | U | (#0011: follows master) |
| rot, csrot, zdrot | s, d, c, z | R D N | R D N | U | – |
| rotm | s, d | R D N | R D N | U | – |
| swap | s, d, c, z | R D N | R D N | U | – |
| scal, csscal, zdscal | s, d, c, z | R D N | R D N | U | #0017, #0026 |
| copy | s, d, c, z | R D N | R D N | U | – |
| axpy | s, d, c, z | R D N | R D N | U | – |
| dot, dotu, dotc | s, d, c, z | R D N | R D N | U | – |
| sdsdot, dsdot | s, d | R D N | R D N | U | – |
| nrm2 | s, d, sc, dz | R D N | R D N | U | #0017 (CBLAS s, d) |
| asum | s, d, sc, dz | R D N | R D N | U | #0017 |
| i?amax | s, d, c, z | R D N | R D N | U | #0017, #0014 |
| abs1 | s, d | R N | R N | U | – |

`rotg` was compared bitwise with the reference for 20000 random and
special operands (`-ffp-contract=off` on both sides; see #0018).

### Level 2

| Routine | Types | F77 | CBLAS | ulmblas_ | Issues |
|---|---|---|---|---|---|
| gemv | s, d, c, z | R D N | R D N | U | #0024, #0025 |
| gbmv | s, d, c, z | R D N | R D N | U | #0024, #0025 |
| symv, hemv | s, d / c, z | R D N | R D N | U | #0024 (hemv) |
| sbmv, hbmv | s, d / c, z | R D N | R D N | U | – |
| spmv, hpmv | s, d / c, z | R D N | R D N | U | – |
| trmv, tbmv, tpmv | s, d, c, z | R D N | R D N | U | #0024, #0025 |
| trsv, tbsv, tpsv | s, d, c, z | R D N | R D N | U | #0024, #0025 |
| ger, geru, gerc | s, d / c, z | R D N | R D N | U | #0024 |
| syr, her | s, d / c, z | R D N | R D N | U | #0019, #0024 |
| spr, hpr | s, d / c, z | R D N | R D N | U | #0024 |
| syr2, her2 | s, d / c, z | R D N | R D N | U | #0019, #0024 |
| spr2, hpr2 | s, d / c, z | R D N | R D N | U | #0024 |

### Level 3

| Routine | Types | F77 | CBLAS | ulmblas_ | Issues |
|---|---|---|---|---|---|
| gemm | s, d, c, z | R D N | R D N | U | #0020, #0024, #0025 |
| symm | s, d, c, z | R D N | R D N | U | #0020, #0022 |
| hemm | c, z | R D N | R D N | U | #0020, #0022 |
| syrk | s, d, c, z | R D N | R D N | U | #0020, #0024, #0025 |
| herk | c, z | R D N | R D N | U | #0020, #0024 |
| syr2k | s, d, c, z | R D N | R D N | U | #0016, #0020, #0024, #0025 |
| her2k | c, z | R D N | R D N | U | #0016, #0020, #0021, #0022 |
| trmm | s, d, c, z | R D N | R D N | U | #0020, #0023, #0024, #0025 |
| trsm | s, d, c, z | R D N | R D N | U | #0020, #0024, #0025, #0027 |

Error handling of all routines (#0019, #0020) was in addition compared
call by call with the reference CBLAS for invalid arguments in both
storage orders (reported position, routine and RowMajorStrg).

Not covered: the x86 SSE, AVX and imported micro kernels were read only
where the source reading led to them (beta = 0, zero tests) and run only
by the cluster test runs; the differential tests ran on arm64 (NEON and
reference kernels).
