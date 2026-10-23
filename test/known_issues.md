# Known Bugs and Open Design Questions

This file is the registry of known bugs of ulmBLAS and of open questions
about its intended behavior.  Issue IDs are stable and never reused.  Tests
refer to issues; `make check` validates these references (see
`test/run-tests.sh`).

Format of an entry:

    Issue #NNNN: Short descriptive title
    Type: BUG | DESIGN
    Status: OPEN | RESOLVED | WONTFIX
    Covered by tests: Tnnnn (kind, CLASS), ...   (or: none)

- `kind` is the kind of test: `numeric` (`test/numeric`), `netlib`
  (`test/F77`, `test/C`) or `header`.
- `CLASS` is the classification of the test: `XFAIL` or `DESIGN`.
  - An OPEN BUG is covered by XFAIL tests that check the expected behavior.
    Once the bug is fixed they report XPASS (a failure of `make check`).
    Then turn them into regular tests and set the issue to RESOLVED.
  - An OPEN DESIGN question is covered by DESIGN tests that record the
    currently observed behavior without making it the specification.  If
    the behavior changes, the test reports FAIL until the recorded behavior
    (or the issue) is updated.

Genuine numerical failures are never registered merely to make the test
suite pass: an issue is added only after the failure has been analyzed and
confirmed, and XFAIL and DESIGN results are listed in every `make check`
summary.

---

## Bugs

Issue #0001: rotg returns z = 1 instead of 1/c if |a| == |b|
Type: BUG
Status: OPEN
Covered by tests: T0111 (numeric, XFAIL)

`ulmBLAS::rotg` (`ulmblas/level1/rot.tcc`) sets `z = 1/c` only if
`|a| < |b|`.  Reference BLAS (`drotg`) uses `|b| >= |a|`, so for `|a| == |b|`
ulmBLAS returns `z = 1` instead of `z = 1/c` (e.g. `a = b = 1`: `z = 1`
instead of `sqrt(2)`).  `r`, `c` and `s` are correct.  Since `z = 1` encodes
the rotation `c = 0, s = 1`, code that reconstructs the rotation from `z`
gets a wrong rotation.  The Netlib test data for `rotg` contains no case
with `|a| == |b|`.  Expected: `z = 1/c` if `|b| >= |a|` and `c != 0`.
Affects `cblas_srotg`, `cblas_drotg` and the corresponding functions of the
other interfaces.

Issue #0003: dgetrf with row major storage and m != n computes wrong results and writes outside the matrix
Type: BUG
Status: OPEN
Covered by tests: T0404 (numeric, XFAIL)

For `CblasRowMajor`, `interfaces/lapack/C/getrf.cc` calls
`ulmBLAS::getrf(n, m, A, ldA, 1, ...)`: the dimensions are those of the
transposed matrix, but the strides `(ldA, 1)` describe the matrix itself.
For square matrices this computes `P*A = L*U` (see #0004).  For `m != n` the
result is not a factorization of `A`; for `m < n` elements beyond the last
row of `A` are overwritten and pivot indices `>= m` are returned (memory
corruption).  Expected: a factorization of the `m x n` matrix that only
accesses the matrix (with semantics according to #0004).

## Design questions

Issue #0002: alpha = 0 with NaN or Inf in operands that reference BLAS does not reference
Type: DESIGN
Status: OPEN
Covered by tests: T0600 (numeric, DESIGN)

If alpha is zero, reference BLAS returns immediately (or only scales by
beta) and does not reference A, x or y.  The BLAS documentation guarantees
this only for some functions (e.g. "B need not be set" for TRMM/TRSM if
alpha is zero) and not for others.  ulmBLAS follows reference BLAS for GEMV,
SPMV, SBMV, HER, HER2, HPR, HPR2, the complex packed and banded functions
and all level 3 functions, but GER, GERU, GERC, SYMV, HEMV, SYR, SYR2, SPR
and SPR2 compute `0*x` and propagate NaN or Inf from the operands into the
result (for all supported precisions; the kernels `ger`, `sylmv`, `helmv`,
`sylr`, `sylr2`, `splr`, `spur`, `splr2`, `spur2` in `ulmblas/level2` have
no early return for alpha = 0).  The test records which functions
propagate NaN.  Open: should ulmBLAS match reference BLAS here?  Note:
beta = 0 (C or y need not be set) is documented and is handled correctly by
all tested functions.

Issue #0004: Semantics of dgetrf with row major storage
Type: DESIGN
Status: OPEN
Covered by tests: T0401 (numeric, DESIGN)

The C interface of LAPACK follows ATLAS (0-based pivots, `CBLAS_ORDER`
argument).  ATLAS' `clapack_dgetrf` factors a row major matrix as the
transposed column major matrix, i.e. `A = L*U*P` with column interchanges
and unit diagonal in `U`.  ulmBLAS computes `P*A = L*U` with row
interchanges for square row major matrices (the result of the call in
`getrf.cc`, see #0003).  The test records the observed semantics ("rows" or
"columns").  Open: which semantics is intended?  The answer determines the
fix of #0003.
