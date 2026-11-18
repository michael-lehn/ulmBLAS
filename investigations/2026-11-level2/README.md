# Level 2 for getrf: gemv, ger, trsv (2026-11)

Roadmap step 5.  Sources read: OpenBLAS 0.3.34, BLIS (BLAS-like Library
Instantiation Software; BLAS: Basic Linear Algebra Subprograms) 2.0.
Measured on heim (i5-12500, 6 cores, GCC (GNU Compiler Collection) 12,
`-march=native`), OpenBLAS 0.3.34 pthreads.

What getrf uses from level 2: getf2 calls `ger` (column major, unit
increments; plus iamax, swap, scal from level 1).  getrf2 is recursive
(gemm, trsm), laswp has no level 2.  gemv and trsv: getrs with one
right-hand side goes through trsm, so for getrf only ger matters; gemv and
trsv are measured anyway (users call them).

## OpenBLAS

Threads (interface/*.c, `GEMM_MULTITHREAD_THRESHOLD` = 4):

| op            | serial if             | threads                      |
|---------------|-----------------------|------------------------------|
| s/dgemv       | m n < 115200 * 4      | all (num_cpu_avail)          |
| c/zgemv       | m n < 1024 * 4        | all                          |
| s/dger        | m n <= 2048 * 4       | 3 up to 20480 * 4, then all  |
| zger          | m n <= 36 * 16^2 * 4  | all                          |
| cger          | m n <= 36 * 8^2 * 4   | all                          |
| trsv          | always                | none                         |

So dgemv: threads from about 680 x 680, zgemv already from 64 x 64.

driver/level2/gemv_thread.c: N (A not transposed) splits the rows of y, T
(A transposed) splits the columns
of A (= elements of y): equal parts, at least 4 per thread, no reduction.
Only N with m < 4 nt (too few rows) splits x and adds per-thread copies of
y (thread-local buffer, m nt <= 1024, m n > 24^2 * 4^2).  beta y is done
before (interface, scal).

driver/level2/ger_thread.c: splits the columns of A, equal parts, at least
4 each.

trsv (driver/level2/trsv_L.c, trsv_U.c): blocks of DTB_ENTRIES (64)
columns, inside a block axpy (N) or dot (T) column by column, then one gemv
kernel call (not the threaded gemv) for the rest of the column panel.

Kernels for HASWELL (also used for Alder Lake):

- dgemv_n_4.c: blocks of NBMAX = 2048 rows of y (16 KB (kilobytes), L1
  (level 1 data cache)); in a block 4 columns per kernel call (kernel_4x4, 4
  FMA (fused multiply-add) per vector of y), then 2 and 1.
  inc_y != 1: y block in a buffer.
- dgemv_t_4.c + dgemv_t_microk_haswell-4.c: blocks of NBMAX = 2048 rows
  (x block 16 KB in L1); 4 columns per call, one ymm (256-bit AVX (Advanced
  Vector Extensions) register) accumulator per
  column, x loaded once for 4 columns, unrolled twice (8 rows per
  iteration, still 4 accumulators: latency bound, about 1 FMA per cycle).
  Columns in groups of 4, results via ybuffer, then y += alpha ybuffer.
- sgemv_t_4.c: NBMAX 4096; zgemv_n/t_4.c: NBMAX 1024, 4 columns.
- dger: axpy per column (kernel/x86_64/dger.c: daxpy kernel).

## BLIS

- gemv: front end chooses the variant from the storage of A: column-stored
  (after transposition) var2 = fused axpyf, row-stored var1 = fused dotxf.
  The unf variants call axpyf/dotxf with b_fuse columns (rows) at a time.
- Fusing factors, haswell config (bli_cntx_init_haswell.c): AF (axpyf) =
  DF (dotxf) = 8 for s and d (zen3: AF 5, DF 8).  ddotxf_zen_int_8: 8 rows of A^T, one
  ymm accumulator each (8 accumulators, x loaded once for 8 FMA), 4 rows
  per iteration.
- ger: unb var1/var2, axpyv per row/column.
- trsv: unf var1/var2 with the fusing factor as block size.
- No threads in level 2 (frame/2 has no threading code).

## ulmBLAS before

- gemv column major: fused axpyf, FuseFactor (config/fusefactor.h); with
  USE_SSE (all x86-64) axpyf for double is the SSE (Streaming SIMD
  Extensions; SIMD: single instruction, multiple data) kernel with fuse factor
  2 (axpy2v, 128-bit); float and complex<float>: fuse factor 1 (axpy per
  column).
- gemv row major (BLAS T): `ref::dotuxf` with 4 rows: scalar loop, 4 sums
  (no vectorization of the reductions without -ffast-math).
- ger: axpy per column (vectorized level 1 axpy).
- trsv: blocks of the fuse factor (4), gemv for the rest.
- No threads.

## Candidates

- gemv T: dotxf-like kernel with F rows and W lanes each (F W accumulators,
  8 vectors; x loaded once for F rows), plain C++ with omp simd (the SIMD
  directive of OpenMP (Open Multi-Processing)) as in
  level1/kernel/reduce.h.  F = 4 or 8, W = 1 or 2 vectors.  Block of x
  (2048-4096 elements) as OpenBLAS.
- gemv N: axpyf-like kernel with F columns (F FMA per load/store of y),
  y in blocks of rows (L1).  F = 4 or 8.
- complex: same loops on the real view; T with a copy of x with real and
  imaginary parts swapped (no permutes in the loop), N with the pair loop
  of zaxpy.
- Threads: split the rows of y in both cases (N: rows of A; T: columns of
  A), no reduction; ger: split columns (column major) or rows.  Thresholds
  as OpenBLAS, constants in one place.
- trsv: OpenBLAS does not thread it.  Larger blocks (64) so that the gemv
  for the rest of the panel uses the fused kernel.

## Findings on the way

- Small problems depend much on the placement of A, x, y (4K aliasing):
  zgemv_t n = 100, one thread, y at different offsets in its page:
  OpenBLAS 13 to 58 GFLOP/s (10^9 floating-point operations per second),
  ulmBLAS 7 to 43.  The first size measured in
  a process was often slow (A from mmap, page aligned).  l2bench.c now
  measures 5 layouts (A, x, y at pseudo-random multiples of 64 bytes from
  page boundaries, the same for all libraries) and prints the median.
  The first baseline (results/openblas.txt, before.txt: one layout from
  malloc) is noisier than the second (openblas2.txt, before2.txt).
- Login sessions on the E44 machines start tracker-extract etc. for a few
  seconds: benchmarks start after `sleep 10` (or nohup).
- Thread pool: an empty parallel region on 6 threads took 1.75 us
  (microseconds), 0.37 us without pinning the calling thread (three system
  calls per region: save, set, restore of its CPU (central processing
  unit) mask).  Not pinning the caller at all was
  worse: placed on the core of a worker (pinned there in the first region)
  dgemv n = 1000 ran at 16-23 instead of 57 GFLOP/s.  Now level 2 pins the
  caller only if sched_getcpu() says it shares a core with a worker
  (threads::callerSharesCore); afterwards it usually stays on its CPU.
- GCC 12 and the lanes of the T kernel: the main loop keeps the F L
  accumulators in registers (8 FMA with memory operands, 2 loads of x per
  iteration), but a loop over the halving widths was not unrolled (lanes
  through the stack, 16-byte stores read as 32 bytes): constant bounds by
  template recursion (Halve), complex: even and odd lanes added in loops
  of constant length.
- A kernel test program with everything inlined into one translation unit
  (l2kern.cc) gave other numbers than the library (sgemv_t n = 100: 30
  against 70): decisions are made with the library through CBLAS (the C
  interface of BLAS).

## What was done

- ulmblas/level2/kernel/gemvf.h: fused kernels, plain C++ with omp simd:
  columns (BLAS N, FuseN = 8 columns per pass for s/d, 4 for c/z),
  rows (BLAS T, FuseT = 4 rows, 8 vectors of partial sums), complex as
  real numbers (N: sums re(a) A and im(a) A, one swap per element of y;
  T: x and swapped x, no permutations in the loop), rank1 for ger.
- ulmblas/level2/gemvfused.h: blocks of 16 KB of y (N) or x (T), the
  kernels for float, double, complex; other types and strides as before.
- Threads (ulmblas/level2/threads.h): gemv splits y (N: rows of A in parts
  of 64 bytes; T: columns of A in parts of FuseT), ger splits the columns
  (parts of 4) or, for tall and skinny A, the rows; thresholds as OpenBLAS.
  trsv: no threads (OpenBLAS has none either).
- trsv: blocks of 64 columns (OpenBLAS DTB_ENTRIES), inside a block
  FuseN columns at a time and the fused kernel for the rest of the block;
  the gemv calls of trsv, trmv run on one thread (gemvSerial).
- All parameters in ulmblas/config/level2.h (macros for make tune).
- ThreadPool::run(nt, f, pinCaller): see above.

## Measurements

heim, OpenBLAS 0.3.34 (pthreads), median of 5 layouts (l2bench.c,
run.sh), column major, lda = m = n, unit increments.  GFLOP/s of OpenBLAS,
then before (30a5e714) and after as ratio to OpenBLAS.

### 1 thread: GFLOP/s openblas2, then before2, after as ratio to it

| op | 100 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|
| dgemv_n | 30.6 / 0.25 / 0.85 | 19.1 / 0.40 / 1.04 | 18.1 / 0.42 / 1.05 | 7.6 / 0.88 / 0.99 | 6.1 / 1.05 / 0.97 | 5.8 / 1.13 / 0.97 |
| dgemv_t | 31.1 / 0.50 / 1.06 | 19.9 / 0.73 / 1.01 | 18.8 / 0.78 / 0.97 | 7.5 / 1.00 / 0.99 | 6.4 / 0.99 / 0.92 | 6.2 / 0.96 / 0.91 |
| sgemv_n | 56.7 / 0.09 / 0.79 | 48.5 / 0.09 / 0.99 | 37.6 / 0.11 / 0.98 | 33.0 / 0.12 / 0.96 | 15.6 / 0.25 / 0.85 | 13.7 / 0.28 / 0.83 |
| sgemv_t | 76.1 / 0.07 / 0.95 | 58.2 / 0.09 / 0.86 | 36.8 / 0.14 / 1.00 | 32.8 / 0.16 / 0.97 | 13.6 / 0.37 / 0.95 | 12.1 / 0.42 / 0.95 |
| zgemv_n | 49.0 / 0.22 / 0.76 | 37.1 / 0.29 / 0.91 | 25.3 / 0.42 / 0.89 | 15.3 / 0.66 / 0.91 | 13.7 / 0.73 / 0.93 | 13.1 / 0.76 / 0.94 |
| zgemv_t | 58.0 / 0.17 / 0.67 | 33.9 / 0.29 / 0.96 | 28.0 / 0.36 / 1.00 | 14.7 / 0.65 / 0.94 | 13.7 / 0.69 / 0.94 | 13.1 / 0.72 / 0.95 |
| zgemv_c | 54.5 / 0.17 / 0.51 | 37.0 / 0.25 / 0.77 | 28.0 / 0.32 / 0.82 | 14.9 / 0.59 / 0.95 | 13.7 / 0.64 / 0.94 | 13.1 / 0.67 / 0.95 |
| dger | 17.1 / 1.00 / 0.98 | 7.3 / 1.01 / 1.06 | 7.2 / 1.03 / 1.04 | 4.8 / 0.93 / 0.96 | 3.3 / 0.97 / 0.99 | 3.1 / 0.98 / 0.98 |
| zgeru | 34.2 / 0.84 / 0.83 | 13.0 / 1.15 / 1.16 | 12.6 / 0.91 / 1.10 | 7.3 / 0.97 / 0.96 | 6.3 / 1.00 / 0.99 | 6.2 / 1.01 / 1.00 |
| dtrsv_ln | 10.1 / 0.65 / 0.72 | 18.0 / 0.39 / 0.91 | 14.9 / 0.48 / 0.97 | 12.1 / 0.55 / 0.82 | 6.2 / 0.95 / 0.98 | 6.0 / 1.01 / 0.95 |
| dtrsv_lt | 8.8 / 1.16 / 0.93 | 18.4 / 0.74 / 0.95 | 14.7 / 0.82 / 1.07 | 10.2 / 0.86 / 1.05 | 6.2 / 0.97 / 0.98 | 6.0 / 0.97 / 0.94 |
| dtrsv_un | 5.8 / 0.92 / 0.96 | 14.8 / 0.44 / 0.98 | 13.5 / 0.53 / 1.01 | 11.1 / 0.58 / 0.94 | 5.9 / 0.99 / 0.96 | 5.8 / 1.03 / 0.95 |
| dtrsv_ut | 10.0 / 1.17 / 1.31 | 19.5 / 0.71 / 1.12 | 15.4 / 0.80 / 1.08 | 9.6 / 1.00 / 1.30 | 6.3 / 0.95 / 0.97 | 6.0 / 0.95 / 0.96 |
| ztrsv_ln | 18.6 / 0.38 / 0.45 | 25.1 / 0.31 / 0.73 | 29.1 / 0.27 / 0.76 | 15.2 / 0.50 / 0.84 | 14.0 / 0.55 / 0.91 | 13.7 / 0.56 / 0.91 |
| ztrsv_lt | 17.2 / 0.47 / 0.86 | 24.8 / 0.38 / 1.04 | 28.0 / 0.35 / 1.05 | 13.5 / 0.65 / 0.97 | 13.1 / 0.67 / 0.94 | 13.2 / 0.69 / 0.94 |
| ztrsv_un | 16.1 / 0.43 / 0.54 | 24.5 / 0.31 / 0.76 | 28.2 / 0.28 / 0.80 | 14.1 / 0.53 / 0.84 | 13.4 / 0.57 / 0.89 | 13.3 / 0.57 / 0.91 |
| ztrsv_ut | 17.3 / 0.47 / 0.89 | 27.1 / 0.36 / 0.93 | 29.1 / 0.34 / 1.05 | 14.0 / 0.65 / 0.96 | 13.2 / 0.68 / 0.95 | 13.4 / 0.68 / 0.93 |
| dgemv_rn | 31.0 / 0.50 / 1.07 | 18.9 / 0.77 / 1.05 | 17.2 / 0.85 / 1.07 | 7.6 / 0.98 / 0.98 | 6.4 / 0.99 / 0.92 | 6.1 / 0.97 / 0.91 |
| dgemv_rt | 30.6 / 0.25 / 0.83 | 19.2 / 0.39 / 1.00 | 17.4 / 0.44 / 1.03 | 7.5 / 0.89 / 0.99 | 6.0 / 1.07 / 0.99 | 5.7 / 1.15 / 0.99 |

### 6 threads: GFLOP/s openblas2, then before2, after as ratio to it

| op | 100 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|
| dgemv_n | 30.7 / 0.25 / 0.85 | 18.7 / 0.40 / 1.08 | 144.3 / 0.05 / 1.06 | 17.8 / 0.37 / 0.98 | 8.4 / 0.76 / 1.00 | 7.5 / 0.87 / 1.00 |
| dgemv_t | 31.3 / 0.50 / 1.05 | 20.2 / 0.71 / 1.01 | 49.7 / 0.29 / 1.55 | 14.6 / 0.52 / 1.04 | 8.0 / 0.79 / 1.01 | 7.6 / 0.80 / 0.98 |
| sgemv_n | 51.7 / 0.10 / 0.87 | 48.5 / 0.09 / 1.01 | 327.5 / 0.01 / 0.99 | 124.9 / 0.03 / 1.31 | 21.0 / 0.19 / 1.00 | 15.2 / 0.25 / 1.02 |
| sgemv_t | 77.2 / 0.07 / 0.94 | 57.6 / 0.09 / 0.93 | 156.7 / 0.03 / 1.02 | 69.4 / 0.08 / 1.17 | 16.4 / 0.31 / 1.08 | 14.9 / 0.34 / 1.02 |
| zgemv_n | 92.6 / 0.12 / 0.87 | 260.2 / 0.04 / 0.82 | 114.0 / 0.09 / 0.90 | 20.4 / 0.50 / 1.00 | 15.3 / 0.65 / 1.00 | 14.5 / 0.69 / 1.00 |
| zgemv_t | 78.6 / 0.12 / 0.50 | 181.7 / 0.06 / 0.68 | 110.8 / 0.09 / 0.83 | 19.1 / 0.50 / 1.02 | 15.2 / 0.62 / 1.02 | 14.0 / 0.67 / 1.00 |
| zgemv_c | 90.7 / 0.10 / 0.53 | 206.6 / 0.04 / 0.58 | 120.4 / 0.08 / 0.59 | 18.9 / 0.47 / 1.02 | 15.2 / 0.58 / 1.01 | 13.9 / 0.63 / 1.00 |
| dger | 29.9 / 0.57 / 0.90 | 91.7 / 0.08 / 0.95 | 55.4 / 0.13 / 0.99 | 7.7 / 0.59 / 1.01 | 4.0 / 0.82 / 0.99 | 3.6 / 0.86 / 0.99 |
| zgeru | 94.8 / 0.30 / 0.77 | 189.7 / 0.08 / 0.80 | 73.4 / 0.16 / 1.17 | 9.3 / 0.76 / 1.00 | 7.5 / 0.84 / 0.97 | 7.1 / 0.88 / 0.98 |
| dtrsv_ln | 9.9 / 0.66 / 0.73 | 18.2 / 0.39 / 0.90 | 15.5 / 0.46 / 0.96 | 8.8 / 0.72 / 1.28 | 6.2 / 0.94 / 0.99 | 6.0 / 1.00 / 0.97 |
| dtrsv_lt | 8.5 / 1.20 / 0.96 | 18.6 / 0.71 / 0.94 | 16.1 / 0.74 / 0.96 | 10.3 / 0.86 / 1.14 | 6.3 / 0.95 / 0.95 | 6.1 / 0.95 / 0.90 |
| dtrsv_un | 5.8 / 0.92 / 0.97 | 14.9 / 0.47 / 0.98 | 14.1 / 0.51 / 1.00 | 11.4 / 0.54 / 0.77 | 5.9 / 0.98 / 0.99 | 5.8 / 1.03 / 0.96 |
| dtrsv_ut | 9.7 / 1.21 / 1.36 | 20.1 / 0.69 / 1.07 | 17.1 / 0.72 / 0.96 | 11.2 / 0.82 / 0.95 | 6.4 / 0.93 / 0.93 | 6.1 / 0.94 / 0.91 |
| ztrsv_ln | 18.4 / 0.39 / 0.45 | 26.0 / 0.30 / 0.73 | 29.1 / 0.27 / 0.78 | 15.8 / 0.48 / 0.83 | 14.1 / 0.54 / 0.91 | 13.8 / 0.56 / 0.91 |
| ztrsv_lt | 17.1 / 0.47 / 0.87 | 27.2 / 0.35 / 0.88 | 28.9 / 0.34 / 0.96 | 13.7 / 0.64 / 0.94 | 13.2 / 0.67 / 0.93 | 13.3 / 0.69 / 0.92 |
| ztrsv_un | 16.0 / 0.43 / 0.54 | 25.9 / 0.30 / 0.76 | 29.4 / 0.27 / 0.78 | 14.1 / 0.53 / 0.84 | 13.4 / 0.57 / 0.89 | 13.3 / 0.58 / 0.91 |
| ztrsv_ut | 17.9 / 0.44 / 0.88 | 29.0 / 0.33 / 0.88 | 29.9 / 0.33 / 0.96 | 14.3 / 0.63 / 0.93 | 13.3 / 0.67 / 0.93 | 13.5 / 0.68 / 0.92 |
| dgemv_rn | 31.1 / 0.47 / 1.06 | 19.9 / 0.72 / 1.00 | 50.1 / 0.29 / 1.50 | 14.8 / 0.50 / 1.01 | 8.1 / 0.79 / 1.01 | 7.6 / 0.78 / 0.98 |
| dgemv_rt | 30.5 / 0.25 / 0.85 | 18.8 / 0.41 / 1.05 | 141.3 / 0.05 / 1.09 | 17.3 / 0.39 / 1.01 | 8.4 / 0.76 / 1.00 | 7.5 / 0.87 / 1.00 |

Raw data: results/openblas2.txt, before2.txt, after.txt (the first
baseline with one layout: openblas.txt, before.txt).

Summary: the real types are on par from n = 500 (0.83-1.55), threads
included; n = 100 0.72-1.36.  dgemv T in cache (the gap of before, 0.5):
1.0-1.06.  Remaining gaps (after / OpenBLAS):

- complex gemv for n <= 1000: zgemv_c 0.51-0.82 (6 threads 0.53-0.59),
  zgemv_t 0.67 at n = 100 (6 threads 0.50-0.83), zgemv_n 0.76-0.91.  The
  same loop for T and C (zgemv_t 1.0 at n = 1000, zgemv_c 0.82): code
  placement?  Variants of FuseN, FuseT, lanes changed little, and changes
  elsewhere in the library moved zgemv_n n = 100 by up to 30 % (code
  alignment of the loops).  Candidates: unrolled loops with the
  permutations of OpenBLAS's zgemv_t kernel, one copy of x for all threads
  (packed by the caller), measurements with a profiler.
- ztrsv with A not transposed: 0.45-0.54 at n = 100, 0.73-0.8 for 500-1000
  (with A transposed on par).  Neither the size of the diagonal blocks (16
  to 128) nor fused columns instead of axpy in them changed it.
- sgemv_n n = 4000, 8000: 0.83-0.85 (memory bound, 1 thread).
- n = 100: sgemv_n 0.79, dgemv_n 0.85, dtrsv_ln 0.72 (one call 0.6-1.2
  us).
