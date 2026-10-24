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
  (`test/F77`, `test/C`) or `header`.  It is required for every listed
  test: a run validates the entries of the kinds of tests it selects (e.g.
  `make check-netlib` does not build the numeric tests and reports their
  entries as not checked; `make check` validates all entries).
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
Status: RESOLVED
Covered by tests: T0110, T0111, T0112 (numeric, regression tests)

`ulmBLAS::rotg` (`ulmblas/level1/rot.tcc`) set `z = 1/c` only if
`|a| < |b|`.  Reference BLAS (`drotg`) uses `|b| >= |a|`, so for `|a| == |b|`
ulmBLAS returned `z = 1` instead of `z = 1/c` (e.g. `a = b = 1`: `z = 1`
instead of `sqrt(2)`; `a = -2, b = 2`: `z = 1` instead of `-sqrt(2)`).
`r`, `c` and `s` were correct.  Since `z = 1` encodes the rotation
`c = 0, s = 1`, code that reconstructs the rotation from `z` got a wrong
rotation.  The Netlib test data for `rotg` contains no case with
`|a| == |b|`.  Fixed by using `|a| <= |b|`; affected `srotg` and `drotg` of
all interfaces.  The complex `rotg` (no `z`) was not affected.

Issue #0003: dgetrf with row major storage computes wrong results and writes outside the matrix
Type: BUG
Status: RESOLVED
Covered by tests: T0401, T0404, T0405 (numeric, regression tests)

For `CblasRowMajor`, `interfaces/lapack/C/getrf.cc` called
`ulmBLAS::getrf(n, m, A, ldA, 1, ...)`: the dimensions of the transposed
matrix, but the strides `(ldA, 1)` of the matrix itself.  For square
matrices this computed `P*A = L*U` with row interchanges instead of the ATLAS
convention (see #0004).  For `m != n` the result was not a factorization of
`A`; for `m < n` elements beyond the last row were read and overwritten and
pivot indices `>= m` were returned (memory corruption, confirmed with
AddressSanitizer).  Fixed by factoring the transposed matrix in column major
storage, `ulmBLAS::getrf(n, m, A, 1, ldA, ...)`; the LU algorithm itself was
correct.  The fix changes the result for square row major matrices to the
ATLAS convention.

Issue #0005: Invalid arguments: computation continued after xerbla
Type: BUG
Status: RESOLVED
Covered by tests: T0700, T0701 (numeric, regression tests); Netlib error-exit tests

188 functions of the C interface (92) and the Fortran 77 interface (96)
reported an invalid argument through `cblas_xerbla` or `xerbla_` and then
computed with the invalid arguments (e.g. `cblas_dsyr` with `n = 4`,
`ldA = 1`), which read and wrote memory outside the operands (confirmed
with AddressSanitizer).  Reference CBLAS and BLAS return from the function
after calling the error handler.  The error handlers of ulmBLAS print the
message and return (reference BLAS stops the program in `XERBLA`, reference
CBLAS calls `exit` in `cblas_xerbla`; both can be replaced by the user), so
the bug affected regular calls, and the Netlib error-exit tests, whose
error handlers return, failed intermittently (memory corruption).  Fixed by
returning after the error handler, as the CBLAS `gemv` functions already
did.  The error handlers still do not terminate the program.

Issue #0006: gescal with general strides propagates NaN and Inf if the scalar is zero
Type: BUG
Status: RESOLVED
Covered by tests: T0343 (numeric, regression tests)

`ulmBLAS::gescal` (`ulmblas/level1extensions/gescal.tcc`) scaled a matrix
with unit row or column stride through `scal`, which assigns zero if the
scalar is zero, but a matrix with general strides by `A(i,j) *= alpha`, so
NaN and Inf were kept for `alpha = 0`.  It computes `beta*C` in the buffered
GEMM micro kernel and in gemm, symm, hemm, trmm and trsm for `alpha = 0` or
`k = 0`; with `beta = 0` (C need not be set) the result then contained NaN.
Only reachable through the C++ interface with general strides of C: the
BLAS interfaces always pass a unit row or column stride.  Fixed by scaling
the columns of a matrix with general strides through `scal` as well.

Issue #0007: SSE GEMM micro kernel: assembler labels are not local on Mach-O (macOS x86-64)
Type: BUG
Status: RESOLVED
Covered by tests: T0340, T0341, T0342 (numeric, regression tests; macOS x86-64 only)

The inline assembly of `ulmblas/level3/ukernel/sse/ugemm.tcc` used named
labels `.D...%=` (`.DLOOP%=`, `.DBETANONZERO%=`, ...).  On Mach-O only
labels starting with `L` are local to the assembler; the labels became
(non-external) symbols in `__text`.  With `.subsections_via_symbols` the
linker splits a section into atoms at every symbol and may assume that
code does not fall through from one atom into the next.  The kernel is
inlined into weak (coalesced) template functions; their copies in
different object files differ, and the linker chose the atoms of the
function and of its labels independently, from different object files.
Forward branches to the labels were emitted as relocations against the
label symbols.  The linked kernel was corrupted: on x86-64 under Rosetta 2
(macOS 26) the programs crashed (SIGSEGV inside the kernel, e.g. dgemm,
dsymm, dsyrk, dsyr2k, dtrmm, T0341, T0342) or computed wrong results
(dgetrf for n > 64, dtrsm); reproduced with a minimal example (two copies
of a weak function with such a label: SIGSEGV).  With ELF (Linux) a weak
function is a COMDAT section that the linker keeps or discards as a whole,
so the labels were harmless.  Fixed by numeric local labels (`1:` with
`1b`, `1f`), which are local to the assembler for all object formats; the
instructions are unchanged (ELF: identical machine code, Mach-O: the
branch displacements are resolved by the assembler).

Issue #0008: SSE GEMM micro kernel reads C if beta = 0
Type: BUG
Status: RESOLVED
Covered by tests: T0340, T0341, T0301, T0309, T0315, T0321 (numeric, regression tests)

The SSE kernel (`ulmblas/level3/ukernel/sse/ugemm.tcc`, double, x86-64)
always computed `beta*C + alpha*A*B`.  With `beta = 0` (C need not be set)
NaN or Inf in C was propagated into the result (`0*NaN` is NaN).  Worse,
the buffered variant of the micro kernel (partial tiles at the edges of
every level 3 operation, also in syrk, syr2k, herk and her2k) calls the
kernel with `beta = 0` on an uninitialized local tile: depending on its
previous contents, dgemm, dsymm, dsyrk, dsyr2k, dtrmm, dtrsm and dgetrf
returned NaN or wrong results for any beta.  Confirmed on x86-64 hardware
(Intel Core i5-12500, Debian 12, GCC 12 and Clang 16) and under Rosetta 2.
Fixed by a `beta == 0` branch in the kernel: the accumulators scaled by
alpha are stored without reading C; the code for `beta != 0` is unchanged.
For `beta = 0` the kernel is faster than before (no loads of C, up to 23%
for `k = 4`); for `beta != 0` the performance is unchanged.

Issue #0009: SSE micro kernels: inline assembly writes memory without telling the compiler
Type: BUG
Status: RESOLVED
Covered by tests: T0340, T0341, T0305, T0327, T0331, T0400 (numeric, regression tests)

The inline assembly of the SSE kernels `ugemm`, `utrlsm` and `utrusm`
(`ulmblas/level3/ukernel/sse/`) writes C (and B) through pointers that are
only input operands, without a `"memory"` clobber.  The compiler may then
assume that the memory is not modified by the assembly, e.g. that the local
tile of the buffered micro kernel still holds the values stored before the
call: GCC 12 and 16 remove the stores and use the old values (shown with a
minimal example).  Fixed by adding `"memory"` to the clobber lists; the
instructions of the kernels are unchanged.  Fixed together with #0008; the
native failures with GCC and beta != 0 (e.g. T0305, T0327, T0331, T0400 to
T0407) were not attributed to #0008 or #0009 individually.

Issue #0010: trlscal and truscal with general strides propagate NaN and Inf if the scalar is zero
Type: BUG
Status: RESOLVED
Covered by tests: T0346 (numeric, regression tests)

Analogous to #0006 (GitHub issue #15): `ulmBLAS::trlscal` and
`ulmBLAS::truscal` (`ulmblas/level1extensions/trlscal.tcc`, `truscal.tcc`)
scaled a triangular matrix with unit row or column stride through `scal`,
which assigns zero if the scalar is zero, but a matrix with general strides
by `A(i,j) *= alpha`, so NaN and Inf were kept for `alpha = 0`.  They scale
the triangle of C by beta in syrk, syr2k, herk and her2k and in the
diagonal tiles of their macro kernels; with `beta = 0` (C need not be set)
the result then contained NaN.  Only reachable through the C++ interface
with general strides of C.  Fixed by scaling the columns (trlscal) or rows
(truscal) of the triangle through `scal` as well.

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
Status: RESOLVED
Covered by tests: T0401, T0404 (numeric, regression tests)

The C interface of LAPACK follows ATLAS (0-based pivots, `CBLAS_ORDER`
argument, also exported with ATLAS names by `make atllapack`).  Answer: the
ATLAS convention.  A row major matrix is factored as its transpose in column
major storage, i.e. `A*P = L*U` with column interchanges, `L` lower
triangular and `U` unit upper triangular; the pivot indices refer to columns.
This is the convention checked by the ATLAS LU tester in `bench/lutst.c`
(`lutestR`), which ulmBLAS failed for every row major matrix before the fix
of #0003 and passes afterwards.
