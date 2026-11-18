# Bounded memory (2026-11)

Rule: every dynamic allocation is bounded by a constant (block sizes of
the configuration, number of threads), never proportional to m, n, k,
nrhs (number of right-hand sides), ...  Trigger: `withUnitStride` copied
the whole m x n matrix if neither stride is 1 (potrf2, getrf2 with the
whole matrix on the serial paths).

Branch bounded-memory (from feature/gemm-redesign 9865a168).

## Audit

Grep over ulmblas/ and interfaces/ (experiments/ not included):
`MemoryPool<T>::allocate`, `Buffer<T>`, posix_memalign, malloc,
_aligned_malloc, std::vector (none in the library), new.

| site | size (elements) | bounded before | after |
|------|-----------------|----------------|-------|
| lapack/unitstride.tcc withUnitStride | m n | no | at most lapackCopyMax = KC NC (gemm block sizes); larger: no copy |
| lapack/getrf2.tcc pivots | min(m,n) | no | removed, pivots stored directly (PivIndex, incPiv) |
| lapack/getrf.tcc pivots (task path) | min(m,n) | no | removed |
| lapack/getrs.tcc pivots | n | no | removed, SolveStep reads PivIndex with increment and base |
| lapack/getri.tcc L | n^2 | no | removed: inv(U), inv(L) in place (trtri), U L in place |
| lapack/tasks/getrftasks.tcc tok | mt nt bytes | no | removed: first element of a tile in the depend clauses |
| lapack/tasks/getrftasks.tcc pinfo | kt | no | removed: atomic minimum |
| lapack/tasks/getrftasks.tcc panel copy (team) | m nb | no | only if m nb <= lapackCopyMax, else the team works in place |
| lapack/tasks/getrftasks.tcc candVal, candIdx, candRow, topRow | 2 size, 2 size, 2 size ib, 2 ib | yes (team <= 8, ib 16) | unchanged |
| lapack/tasks/potrftasks.tcc tok | nt^2 bytes | no | removed (tile addresses) |
| lapack/tasks/inversetasks.tcc tok (trtri, lauum) | nt^2 bytes | no | removed (tile addresses) |
| lapack/tasks/solvetasks.tcc tok | mt nt bytes | no | removed (tile addresses of B) |
| lapack/buffer.h | wrapper of MemoryPool | - | - |
| interfaces/lapack/F77/solve.cc Pivots (getrs, getri) | n | no | removed: 1-based IPIV passed with pivot base 1 |
| level3/gemm/frame.h serial | MCmax KCmax + MR, KCmax NCmax + NR | yes | unchanged |
| level3/gemm/frame.h threads | 2 (KC NC + NR) jcWays, (MC KC + MR) nt, flags 2 P jcWays | yes (blocks x threads) | unchanged |
| level3/gemm/frametr.h | MCmax KCmax + MR, KCmax NCmax + NR | yes | unchanged |
| level3/gemm/frametrsm.h | max(MCmax,KB) KB + MR, KB NCmax + NR, KB MR + MR | yes | unchanged |
| level3/gemm/frametrmm.h | max(MCmax,KB) KB + MR, KB NCmax + NR | yes | unchanged |
| level3/i8gemm/driver.h | MC KC + (MC/MR) XA, KC NC + (NC/NR) XB | yes | unchanged |
| level3/ukernel/simd/small.h gemm, cgemm (copy of A, row stride != 1) | m k | no (k unbounded) | at most 2^16: k in blocks, m or n > 2^16 to the fallback |
| level3/ukernel/sme/small.h dgemm | 64 k, m k, k n, m n | no (k; n for m <= 1024) | run: k in blocks of 2048, C copies <= 2^18, A and B never copied (else fallback) |
| level2/gemvfused.h | BlockN; 2 min(n, BlockT) | yes | unchanged |
| level2/gemv.tcc | MC NC, NC, MC | yes | unchanged |
| level2/ger.tcc | MC, NC, MC NC | yes | unchanged |
| auxiliary/threadpool.h | 2 pointers per started thread | yes | unchanged |
| auxiliary/memorypool.tcc | ThreadCache (Windows), the blocks | yes | + memoryPoolLargest (test hook) |
| interfaces/blas/C/gemm_s16s16s32.cc | m, n, m min(n,1024) | no | blocks of 1024 x 1024 |
| interfaces/blas/C/gemm_s8u8s32.cc | 4 m, max(m,n), m min(n,1024) (int32 and int64) | no | blocks of 1024 x 1024 |

The SME (Scalable Matrix Extension) change is only syntax-checked (no
run on the M5).

## LAPACK (Linear Algebra PACKage)

potrf2 and getrf2 are recursive.
The copy moved from the top into the recursion: a subproblem with neither
stride 1 is copied if it fits (m n <= lapackCopyMax), otherwise it is
split and trsm/herk/gemm/swaps work on the original strides (the level 3
frames pack into bounded buffers).  Up to lapackCopyMax elements nothing
changes (one copy of everything, as before).  Every caller is covered:
serial potrf and getrf, the potrf tiles, the getrf panels (m x nb).  One
stride 1: no copy, as before.  No separate right-looking serial potrf: the
recursion already copies only bounded diagonal blocks.

lapackCopyMax = KC NC of BlockSize<T>: 1044480 doubles (8 MB, megabytes)
on AVX2 (Advanced Vector Extensions 2; KC 256, NC 4080);
ULMBLAS_LAPACK_COPY_MAX overrides it.  Test configurations with tiny
blocks (testcase01: 144 elements) run the strided recursion almost
everywhere.

getri: inv(A) = inv(U) inv(L) P without work space.  inv(U) and inv(L)
(unit) with trtri in place (disjoint triangles), then X = U L in place,
recursive (getriimpl.h):

    [U11 U12] [L11    ]   [U11 L11 + U12 L21  U12 L22]
    [    U22] [L21 L22] = [U22 L21            U22 L22]

rec(A11), A11 += A12 A21 (gemm), A12 := A12 L22 (trmm), A21 := U22 A21
(trmm), rec(A22); base case 32 unblocked (row/column i needs only
indices >= i).  4/3 n^3 flops (floating-point operations) as LAPACK's
dgetri.  With OpenMP (Open Multi-Processing) the product as tasks on tiles
(inversetasks.tcc, ulTasks); first version without tasks: 6 threads 0.8
to 28 GFLOPS (billion floating-point operations per second; gemm/trmm
of the thread pool after the OpenMP regions of trtri).  Neither stride 1 and n^2 <= lapackCopyMax: all on a copy.

Pivots: getrf2, getrfTasks, getrs (SolveStep) and getri take the user's
pivot array (PivIndex, increment) directly; getrs and getri have a
trailing pivBase argument (default 0), the Fortran interface passes 1.

Task dependencies: a char per tile (nt^2 bytes) replaced by the address of
the tile's first element.

## Test

T0385 (test/numeric/boundedmemory.cc): memoryPoolLargest() (largest
request since the last reset, all threads and element types) for potrf,
getrf, getri, tall getrf, small-matrix gemm with long k, the integer GEMM
(general matrix multiply) interfaces, with general strides and matrices
larger than the bound; one thread of the pool.  Bound = largest block of
level 2, level 3 and integer GEMM on moderate problems (calibrated per
configuration), lapackCopyMax, 2^16 complex doubles.  T0386: integer GEMM
interfaces with more than one block of rows and columns against loops.

First version of T0385 calibrated only at n = 300: false failure in
testcase04/06/10 (28 MB, the TrsmFrame of a plan with NC = 13824 chosen
for conjugated trsm at larger sizes; bounded, KB NCmax).  Now each case
also runs column major and the bound is max(bound, largest column major).

Runs (daucher, krafft; GCC 12, ARCHFLAGS=-march=native), 9 of 11 test
configurations (testcase08 neon, testcase09 sme not supported):
check-numeric 0 FAIL (OPENMP=0 and OPENMP=1), check-netlib 0 FAIL,
check-reference-lapack-lin GE, PO, TR 4 PASS (needs
`LIN_LDLIBS="-Wl,--no-as-needed -lstdc++ -lpthread"` on Debian: LDLIBS
goes before the objects).

## Benchmarks

bench.cc, run.sh.  krafft (i5-12500, 6 cores, GCC (GNU Compiler
Collection) 12, -march=native), best of 3 (2 for n = 4000), GFLOPS, before
/ after.  s = row
stride (column stride s n); s = 1 column major.  1 thread: serial build;
6 threads: OPENMP=1 (tasks) and 6 threads of the pool.  OpenBLAS: 1
thread ~/ulmbench/libopenblas.so (0.3.34, single threaded), 6 threads
Debian's openblas-pthread 0.3.21; only s = 1.

Unit stride: unchanged (within noise), getri faster.  General strides:
unchanged up to n^2 <= lapackCopyMax (n <= 1021 for double); above, the
recursion works on the strides: serial potrf/getrf s = 2 up to 17 %
slower, s = 42 about 33 % slower (n = 2000).  6 threads: potrf/getrf
unchanged (the tile tasks never copied the whole matrix); getri s = 2,
n >= 2000: 10-15 % slower, s = 42 n = 2000: 32 % slower.

potrf, 1 thread:

| n | s = 1 | s = 2 | s = 42 | OpenBLAS s = 1 |
|---|-------|-------|--------|----------------|
| 500 | 38.3 / 39.0 | 29.7 / 32.9 | 17.8 / 17.0 | 34.6 |
| 1000 | 42.8 / 42.9 | 37.5 / 37.8 | 20.4 / 20.4 | 47.8 |
| 2000 | 49.5 / 49.5 | 44.8 / 44.0 | 30.5 / 20.9 | 53.6 |
| 4000 | 48.2 / 48.1 | 46.3 / 41.2 | - | 57.2 |

potrf, 6 threads:

| n | s = 1 | s = 2 | s = 42 | OpenBLAS s = 1 |
|---|-------|-------|--------|----------------|
| 500 | 105.1 / 103.9 | 37.2 / 40.4 | 39.1 / 37.9 | 16.8 |
| 1000 | 198.5 / 185.2 | 176.3 / 173.2 | 25.5 / 25.5 | 139.7 |
| 2000 | 242.3 / 243.8 | 165.2 / 167.6 | 22.9 / 23.1 | 222.6 |
| 4000 | 227.7 / 226.0 | 132.5 / 133.0 | - | 238.1 |

getrf, 1 thread:

| n | s = 1 | s = 2 | s = 42 | OpenBLAS s = 1 |
|---|-------|-------|--------|----------------|
| 500 | 39.7 / 39.5 | 35.0 / 35.2 | 15.3 / 15.3 | 38.9 |
| 1000 | 47.6 / 47.4 | 40.9 / 41.0 | 21.3 / 21.4 | 49.4 |
| 2000 | 52.8 / 52.9 | 47.6 / 44.4 | 32.0 / 21.1 | 54.5 |
| 4000 | 56.7 / 56.7 | 53.4 / 44.6 | - | 57.9 |

getrf, 6 threads:

| n | s = 1 | s = 2 | s = 42 | OpenBLAS s = 1 |
|---|-------|-------|--------|----------------|
| 500 | 72.7 / 75.0 | 59.7 / 55.5 | 24.4 / 23.8 | 96.1 |
| 1000 | 170.0 / 170.9 | 129.6 / 131.6 | 20.6 / 20.7 | 198.6 |
| 2000 | 216.3 / 211.8 | 138.7 / 137.2 | 20.4 / 20.4 | 269.4 |
| 4000 | 243.6 / 245.4 | 148.9 / 151.9 | - | 305.5 |

getri (inverse from the factorization), 1 thread:

| n | s = 1 | s = 2 | s = 42 | OpenBLAS s = 1 |
|---|-------|-------|--------|----------------|
| 500 | 40.5 / 45.8 | 38.1 / 43.5 | 16.8 / 24.0 | 35.5 |
| 1000 | 49.1 / 49.5 | 45.3 / 46.0 | 19.4 / 30.0 | 44.1 |
| 2000 | 44.3 / 53.3 | 40.1 / 46.8 | 20.8 / 19.6 | 46.8 |
| 4000 | 46.4 / 53.2 | 40.3 / 43.7 | - | 47.8 |

getri, 6 threads:

| n | s = 1 | s = 2 | s = 42 | OpenBLAS s = 1 |
|---|-------|-------|--------|----------------|
| 500 | 127.5 / 114.1 | 94.4 / 100.6 | 32.0 / 39.4 | 109.8 |
| 1000 | 204.0 / 229.2 | 147.6 / 164.9 | 27.7 / 57.6 | 146.3 |
| 2000 | 225.8 / 234.8 | 150.7 / 128.6 | 27.6 / 18.9 | 154.1 |
| 4000 | 175.6 / 196.1 | 123.9 / 112.0 | - | 150.7 |

## Open

- Strided potrf/getrf/getri above lapackCopyMax: 10-35 % slower (level 3
  on strided memory instead of one copy).  Accepted (OpenBLAS has no
  general strides at all); TODO.md: optional interface with work space
  from the caller and a work space query for whole-matrix copies.
- SME small kernel: run on the M5.
- getri, 6 threads, n = 500: 114 instead of 128 GFLOPS (three parallel
  regions: two trtri, the product).
