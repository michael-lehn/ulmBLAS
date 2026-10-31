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
  (`test/F77`, `test/C`), `header` or `example` (`examples/*.cc`).  It is required for every listed
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

Issue #0011: drotm and drotmg are empty functions; srotm, srotmg, sdsdot, dsdot, crotg, zrotg, csrot, zdrot, scabs1, dcabs1 are missing
Type: BUG
Status: RESOLVED
Covered by tests: T0113, T0114, T0115, T0116, T0117, T0118, T0119 (numeric, regression tests); T0001, T0002, T0013 (netlib, regression tests)

`drotm` and `drotmg` of all interfaces (`cblas_`, `ulmblas_`, Fortran 77
`drotm_`, `drotmg_`) were defined with empty bodies, originally for linking
the ATLAS benchmarks: a call returned without computing anything and
without setting the flag of `drotmg`, so a Fortran program linked with
`libf77blas.a` got wrong results silently.  `srotm`, `srotmg`, `sdsdot` and
`dsdot` (documented as not implemented) and the classic reference BLAS
routines `crotg`, `zrotg`, `csrot`, `zdrot`, `scabs1` and `dcabs1` were
missing.  The Netlib test programs in the repository (`sblat1`, `dblat1` of
the original BLAS distribution) did not test any of them.  Implemented for
all interfaces (portable code in `ulmblas/level1/rot.tcc` and
`sdsdot.tcc`); `sblat1` and `dblat1` were replaced by the versions of LAPACK
3.12.1, which test `?rotm`, `?rotmg` and `sdsdot`/`dsdot`.  `rotmg` includes
the correction of the rescaling of Reference-LAPACK after 3.12.1 (3.12.1
resets h12 to 1 if both scale factors are rescaled).

Issue #0012: Fortran 77 interface calls XERBLA without the length of the routine name
Type: BUG
Status: RESOLVED
Covered by tests: T0013 (netlib, regression test)

The 96 functions of the Fortran 77 interface that check their arguments
called `xerbla_(name, &info)` without the hidden length argument that
Fortran passes for a `CHARACTER*(*)` dummy argument.  The `XERBLA` of the
reference BLAS and of the BLAS test programs of LAPACK 3.12 declares
`CHARACTER*(*) SRNAME` and read an undefined length (the test programs of
LAPACK 3.12.1 reported a wrong routine name for DGEMM and others; with a
large length `XERBLA` reads beyond the name).  The old Netlib test programs
declare `CHARACTER*6` and were not affected.  Fixed by passing the length
(6); the `xerbla_` of ulmBLAS prints the name with this length.

Issue #0013: nrm2 returns NaN for two infinite elements and scnrm2/dznrm2 Inf for (Inf, NaN)
Type: BUG
Status: RESOLVED
Covered by tests: T0120 (numeric, regression test)

The scaled sum of squares of `ulmBLAS::nrm2` computed `(|x_i|/scale)^2` with
`scale = |x_i| = Inf` for a second infinite element: Inf/Inf = NaN instead
of Inf (all four types).  The overload for complex vectors was not declared
in `nrm2.h`, so `scnrm2` and `dznrm2` used the generic loop with
`std::abs` of complex elements, which is Inf for (Inf, NaN) (and
n = 1 had a shortcut with the same result); the reference BLAS (LAPACK
3.10 or newer) returns NaN if a real or imaginary part is NaN and
otherwise Inf if one is infinite.  Found with the test programs of LAPACK
3.12.1.  Fixed by adding one for an element equal to the current scale and
by declaring the complex overload, which treats real and imaginary parts as
separate elements as the reference BLAS does.

Issue #0015: offsets of elements overflow int for large leading dimensions and increments
Type: BUG
Status: RESOLVED
Covered by tests: T0356 (numeric, regression test)

The C, Fortran 77 and LAPACK interfaces instantiated the C++ templates
with the index type of their arguments, `int`.  The templates compute the
offset of an element as a product such as `j*ldC`, also in `int`, which
overflows if the matrix or vector spans more than `INT_MAX` elements,
although every argument fits into an `int` (e.g. `ldC` = 2^30 + 7 with
three columns).  The interfaces now call the templates with
`IndexType = std::ptrdiff_t`.  The index type of the interfaces is
`ULMBLAS_INDEX_TYPE` (default `int`, see INSTALL.md); the test configurations
`testcase10` and `testcase11` use 64-bit index types.

Issue #0016: ?syr2k and ?her2k read B with the leading dimension of A
Type: BUG
Status: RESOLVED
Covered by tests: T0320, T0321, T0322, T0323, T0324, T0325 (numeric, regression test)

`sylr2k`, `syur2k`, `helr2k` and `heur2k` in `ulmblas/level3` packed B
with the row and column strides of A.  For ldB != ldA the results were
wrong, and for ldA > ldB elements outside B were read (all types, all
interfaces).  Neither the tests of ulmBLAS nor the Netlib test programs
noticed it: both use ldB = ldA.  Fixed; the level 3 tests now use
different leading dimensions for A, B and C.

Issue #0017: level 1 functions with increments <= 0 accessed memory outside the vector
Type: BUG
Status: RESOLVED
Covered by tests: T0900, T0901 (numeric, regression test), T0013 (netlib, regression test)

The reference BLAS returns 0 from `?asum` and `i?amax` and leaves x
unchanged in `?scal`, `csscal`, `zdscal` if incx <= 0 (the reference
CBLAS: the same, `i?amax` returns 0 also for n <= 0).  ulmBLAS used x[0]
n times for incx = 0, and for incx < 0 it read (`?asum`, `cblas_i?amax`,
`cblas_snrm2`, `cblas_dnrm2`) or wrote (`?scal`) the n-1 elements in front
of x.  `i?amax` of the Fortran 77 interface returned an index in reverse
order for incx < 0, `cblas_i?amax` returned -1 for n <= 0.  Fixed: the
Fortran 77 and CBLAS functions follow the reference for incx <= 0;
`cblas_?nrm2` supports negative increments as LAPACK 3.12.1.  The
`ulmblas_` functions keep the general increments of ulmBLAS: for incx < 0
the BLAS storage order (as all other `ulmblas_` level 1 functions), for
incx = 0 the element x[0] repeated; `ulmblas_i?amax` returns a 0-based
index and -1 for n <= 0.

Issue #0018: ?rotg overflows for large operands
Type: BUG
Status: RESOLVED
Covered by tests: T0903 (numeric, regression test), T0013 (netlib, regression test)

`srotg` and `drotg` used the algorithm of the reference BLAS before LAPACK
3.10 (scale |a| + |b|), which gives NaN for |a| + |b| > max (e.g.
drotg(1e308, 1e308)), for infinite operands and other signs of zero than
LAPACK 3.12.1.  `crotg` and `zrotg` returned r = Inf, c = 0, s = 0 if
|a|^2 + |b|^2 overflows.  Fixed: both follow DROTG and ZROTG of LAPACK
3.12.1 (Anderson, "Algorithm 978: Safe Scaling in the Level 1 BLAS").
Remaining differences to a gfortran build of the reference are rounding
from fused multiply-adds (gfortran contracts by default) and, for infinite
complex operands, NaN that gfortran produces in the division of a complex
by a real number.

Issue #0019: cblas_?syr, ?syr2, ?her, ?her2 accepted ldA = 0 for n = 0
Type: BUG
Status: RESOLVED
Covered by tests: T0702 (numeric, regression test)

The wrappers checked `ldA < n` instead of `ldA < max(1,n)` (reference:
error 8, or 10 for ?syr2 and ?her2).  Fixed.

Issue #0020: Wrong parameter numbers in messages of cblas_xerbla for row major calls
Type: BUG
Status: RESOLVED
Covered by tests: T0703 (numeric, regression test)

For row major calls the reference CBLAS reports the position of the
argument in the call of the column major routine, sets RowMajorStrg and
cblas_xerbla translates the position (e.g. gemm 4 <-> 5, 9 <-> 11).  The
level 3 wrappers of ulmBLAS did not set RowMajorStrg, so a cblas_xerbla of
the user (or of the Netlib test programs) could not translate, and the
default cblas_xerbla of ulmBLAS did not translate at all.  Fixed: all
wrappers set RowMajorStrg before calling cblas_xerbla, the default
handler translates as the reference.

Issue #0021: cblas_cher2k and cblas_zher2k wrote to alpha for row major storage
Type: BUG
Status: RESOLVED
Covered by tests: T0902 (numeric, regression test)

The row major branch conjugated alpha in the caller's memory and restored
it afterwards, which crashes for an alpha in read-only memory and is a
data race if threads share alpha.  Fixed: a local copy is conjugated, as
in the reference CBLAS.

Issue #0022: cblas.h declared input arguments of ?symm, ?hemm and ?her2k as non-const
Type: BUG
Status: RESOLVED
Covered by tests: T0902 (numeric, regression test: compiles only with const declarations)

B of `cblas_?symm`, alpha, beta and B of `cblas_?hemm` and alpha of
`cblas_?her2k` (and of the `ulmblas_` functions) were declared without
const (CBLAS: `const void *`), so C and C++ callers could not pass
pointers to const.  Fixed; the ABI is unchanged.

Issue #0023: trmm propagates Inf and NaN of B into rows that do not depend on them
Type: BUG
Status: OPEN
Covered by tests: T0603 (numeric, XFAIL)

`trmm` packs the diagonal blocks of A including the zeros of the other
triangle (`trlpack`, `trupack`) and the micro kernel multiplies them with
B, so Inf in a row of B gives NaN (0*Inf) in the other rows of the same
block of MR rows, e.g. dtrmm(L, L, N, N, A = [1 *; 1 1], B = (1, Inf))
gives (NaN, Inf) instead of (1, Inf).  The reference BLAS multiplies only
by elements of the triangle.  trsm is not affected.  A fix changes the
packing or the micro kernel calls of the diagonal blocks.

## Design questions

Issue #0002: alpha = 0 with NaN or Inf in operands that reference BLAS does not reference
Type: DESIGN
Status: RESOLVED
Covered by tests: T0600 (numeric, regression test)

If alpha is zero, reference BLAS returns immediately (or only scales by
beta) and does not reference A, x or y.  The BLAS documentation guarantees
this only for some functions (e.g. "B need not be set" for TRMM/TRSM if
alpha is zero) and not for others.  ulmBLAS followed reference BLAS for
GEMV, SPMV, SBMV, HER, HER2, HPR, HPR2, the complex packed and banded
functions and all level 3 functions, but GER, GERU, GERC, SYMV, HEMV, SYR,
SYR2, SPR and SPR2 computed `0*x` and propagated NaN or Inf from the
operands into the result.  Answer: as reference BLAS.  The kernels `ger`,
`gerc`, `sylr`, `sylr2`, `splr`, `spur`, `splr2` and `spur2` in
`ulmblas/level2` return immediately for alpha = 0, `sylmv` and `helmv`
after `y = beta*y`.  The test checks all functions for both storage orders
and both triangles.

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

Issue #0014: i?amax with NaN
Type: DESIGN
Status: RESOLVED
Covered by tests: T0601 (numeric, regression test)

`i?amax` of ulmBLAS returned the same indices as the reference BLAS up to
LAPACK 3.12.1: an element was selected if its absolute value (|re| + |im|)
was greater than the current maximum, so a NaN was only returned if it was
the first element.  The development version of Reference-LAPACK (after
3.12.1, commit b35878842c) returns the position of the first NaN, and its
test programs check this.  Answer: as the development version.  For real
types the result is the first NaN, otherwise the first maximum.  For
complex types it is the first element with NaN in the real or imaginary
part; otherwise the first element with Inf; otherwise the first maximum of
|re| + |im|, compared as |re|/4 + |im|/4 once the sum overflows.

Issue #0024: products with zero elements that the reference BLAS skips
Type: DESIGN
Status: RESOLVED
Covered by tests: T0602 (numeric, regression test)

Several reference routines skip an operation if an element of x, y, A or
B is zero (e.g. `IF (X(J).NE.ZERO)` in DTRMV, DTRSV, DSYR, DGER, DSYRK,
DTRMM, DTRSM), so NaN or Inf in the corresponding column of the other
operand does not reach the result; others (DGEMV, DGBMV, DHEMV) never
skip.  The level 1 kernels of ulmBLAS returned from axpy for a zero
scalar, so gemv, gbmv and hemv dropped NaN and Inf of A for
alpha*x(j) = 0, ger tested alpha*x(i) (ldA = 1) or alpha*y(j) instead of
y(j), spr and hpr alpha*x(j) instead of x(j), tbsv and tpsv the quotient,
and row major hpr2 each of its two terms.  Answer: the reference
semantics routine by routine, but no data-dependent skipping of zero
elements.  Documented quick returns (alpha = 0, k = 0 with beta = 1,
n = 0, ...) are kept, also the one of ?axpy (alpha = 0), which is now
tested in the interfaces instead of the kernels; ulmBLAS no longer skips
where the reference computes.  Where the reference skips data-dependently
(DTRMV, DTRSV, DTBMV, DTPMV, DTBSV, DTPSV with x(j) = 0; DGER with
y(j) = 0; DSYR, DSYR2, DSPR, DSPR2, DHER, DHER2, DHPR, DHPR2; DSYRK,
DHERK, DSYR2K, DHER2K, DTRMM, DTRSM), ulmBLAS computes the products and
propagates NaN and Inf as IEEE arithmetic does (also -0 becomes +0 in the
transposed tr/tb/tp multiplies); `make check-conformance` lists these
differences as known.  For k = 0 and alpha = NaN or Inf, gemm, syrk, herk,
syr2k, her2k return beta*C (the reference gives NaN in its transposed
branches only).

Issue #0025: ATLAS extensions of the transposition argument
Type: DESIGN
Status: RESOLVED
Covered by tests: T0604, T0704 (numeric, regression test)

The C interface accepts `AtlasConj` (114, conjugate without transposition)
for gemv, gbmv, gemm, trmv, tbmv, tpmv, trsv, tbsv, tpsv, trmm, trsm and
the real ?syrk, ?syr2k, the Fortran 77 interface TRANS = 'R' for the same
routines; the reference reports an error.  Answer: documented extensions
of ulmBLAS (cblas.h, ARCHITECTURE.md); real types compute as without
transposition, complex types with conj(A).  All other values of the
transposition arguments are rejected as before (T0704).

Issue #0026: ?scal with alpha = 0 and NaN or Inf in x
Type: DESIGN
Status: RESOLVED
Covered by tests: T0605 (numeric, regression test), T0013 (netlib, regression test)

`?scal`, `csscal` and `zdscal` set x to zero for alpha = 0; the reference
multiplies, so NaN and Inf give NaN.  Answer: as the reference.  The
public functions (Fortran 77, CBLAS and `ulmblas_` names) use `scalBlas`,
which leaves x unchanged for alpha = 1 and multiplies otherwise (complex
products as in Fortran); the internal `scal`, used for beta = 0 in level
2 and 3, still assigns zeros (issues #0006, #0010).

Issue #0027: trsm with side = Left multiplies by the reciprocals of the diagonal
Type: DESIGN
Status: OPEN
Covered by tests: T0606 (numeric, DESIGN)

The packing of the triangular blocks stores 1/a(j,j); the reference
divides by a(j,j) for side = Left (for side = Right it also uses
1/a(j,j)).  Results differ by rounding, and 1/a(j,j) can overflow where
the quotient does not: dtrsm with a = 1e-310, b = 1e-300 gives Inf
(reference: 1e10).  Open: accept, or divide in the solve kernels?

Issue #0028: types of cblas.h differ from the CBLAS of LAPACK 3.12
Type: DESIGN
Status: OPEN
Covered by tests: T0607 (numeric, DESIGN)

Complex arguments are `float *` / `double *` (CBLAS: `void *`), so C
callers pass `float _Complex *` only with a cast; `i?amax` returns
`ULMBLAS_INDEX_TYPE` (reference: `CBLAS_INDEX`, `size_t`); there is no
`CBLAS_LAYOUT`.  Changing the pointer types to `void *` keeps the ABI and
existing callers.  Open: adopt the reference declarations?

Issue #0029: the ulmblas_ functions do not check their arguments
Type: DESIGN
Status: OPEN
Covered by tests: T0608 (numeric, DESIGN)

The `ulmblas_` names of the C interface (ulmBLAS's own column major API)
call no error handler: invalid dimensions, leading dimensions and
increments are used as given, unknown enumeration values select the else
branch (e.g. Trans for an unknown transposition of real gemv).  Open:
document as unchecked interface (as the C++ templates), or check?
