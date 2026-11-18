# getrf: OpenBLAS, libflame, ulmBLAS (roadmap step 6)

Notes for the LU (lower upper) factorization with partial pivoting on top
of the level 3 frame (feature/gemm-redesign 25013480).

## OpenBLAS 0.3.34

interface/lapack/getrf.c

- threads: 1 if m n < 10000 (double; 40000 single), else num_cpu_avail,
  reduced to m n / 10000 (about 10000 elements per thread: all threads
  from n = 250 on)
- buffer sa (GEMM_P x GEMM_Q), sb behind it
- one thread: getrf_single, otherwise getrf_parallel (pthreads build:
  getrf_parallel.c; USE_OPENMP=1: getrf_parallel_omp.c)

lapack/getrf/getrf_single.c

- blocking = min(min(m,n)/2 rounded up to GEMM_UNROLL_N, GEMM_Q)
  (HASWELL: d Q 256, UNROLL_N 8; z Q 192, UNROLL_N 2); blocking <=
  2 UNROLL_N: getf2
- right-looking, panel of blocking columns factored by a recursive call
  (the same routine on the panel: blocking halves again ... down to getf2)
- per panel: L11 packed once as trsm "A" (TRSM_ILTCOPY into sb); for the
  trailing columns in blocks of REAL_GEMM_R = GEMM_R - max(P,Q), per
  UNROLL_N columns: LASWP_PLUS (interchanges of this panel), GEMM_ONCOPY
  into sbb (packed U), TRSM_KERNEL_LT on the packed columns (result written
  to A, kept packed); then A21 in GEMM_P row blocks (GEMM_ITCOPY) times the
  packed U (GEMM_KERNEL_N)
- fused laswp + trsm + pack: U(K, :) packed once per panel and column
  block, no separate passes over the trailing columns for swaps and trsm
- interchanges to the left at the end (LASWP_PLUS per panel)

lapack/getf2/getf2_k.c

- left-looking (Crout): per column j: the earlier interchanges applied to
  the column, U part with DOTU_K, the rest with GEMV_N (j columns), IAMAX,
  row interchange left of j (SWAP_K), SCAL

lapack/getrf/getrf_parallel.c (pthreads)

- init_bk = min(min(m,n)/2 rounded to UNROLL_N, GEMM_Q); the first panel
  by getrf_parallel itself (recursive, its updates threaded), the later
  panels by getrf_single (one thread)
- lookahead of one panel: the calling thread updates the first `width`
  columns of the trailing matrix (inner_basic_thread: laswp + trsm + gemm,
  all rows), then factors the next panel (getrf_single); the other threads
  update the rest (inner_advanced_thread) at the same time
- inner_advanced_thread: 2D (two-dimensional) split, range_M rows x
  range_N columns per thread; each thread packs its own columns of U
  (laswp + ONCOPY + TRSM_KERNEL fused, DIVIDE_RATE = 2 buffers) and shares
  them through job[].working flags (GotoBLAS gemm threading); each thread
  updates its rows with the U panels of all threads, its rows of L packed
  once per GEMM_P block
- width of the calling thread: FORMULA1 = (n_rem + GETRF_FACTOR m b (1 -
  T) / (b + m)) / T (GETRF_FACTOR 1.0); if width < bk the next panels get
  narrower: FORMULA2 = (n - is + bk) (1 - sqrt(1 - 1/T))
- no barrier per step: the caller waits for the flags of the threads
  before it packs the next L11
- interchanges to the left at the end with blas_level1_thread(LASWP_PLUS)
- interface/lapack/laswp.c: threaded from n = 64 on (columns split)

Takeaway: one serial panel per step (recursive, the single version),
lookahead depth 1, panel width adaptive (FORMULA2), the trailing update
with shared packed U and fused laswp + trsm.

## libflame (FLA_LU_piv)

- lapack2flame dgetrf_ -> FLA_LU_piv -> FLA_LU_piv_internal with the
  control tree fla_lu_piv_cntl2 (its initialization not read)
- blocked variant 5 (right-looking): LU of [A11; A21] with the sub
  control tree, pivots applied to [A10; A20] and [A12; A22], trsm, gemm;
  variants 3 and 4 (left-looking, Crout) and unblocked (optimized)
  variants 3, 4, 5 also exist
- block size from FLA_Query_blocksize as for FLA_Chol; parallel only
  inside BLIS (BLAS-like Library Instantiation Software; BLAS: Basic
  Linear Algebra Subprograms), or SuperMatrix with FLASH objects (not the
  LAPACK (Linear Algebra PACKage) interface)

Takeaway: two levels of blocking, threads only in BLAS (as LAPACK).

## ulmBLAS (25013480)

- getrf.tcc: OPENMP=1 and outside of a parallel region with min(m,n) >
  150: task graph (tasks/getrftasks.tcc: tiles nb 48..128, panel teams of
  up to 8 threads, priorities as PLASMA (Parallel Linear Algebra Software
  for Multicore Architectures)); otherwise getrf2
- getrf2: recursive halving (as LAPACK dgetrf2), base getf2 at n <= 16
  (32 for m <= 96); swapRows serial, column by column; trlsm and gemm on
  the frame's threads (pool); getf2 right-looking (iamax, swap, scal, ger;
  ger threaded for large m n)
- laswp.tcc: serial, blocks of 32 columns
- default build (no OpenMP): only getrf2, i.e. threads only inside trsm,
  gemm and ger

## What maps onto our pool

Candidates:

- lookahead on the pool (as getrf_parallel): thread 0 factors the next
  panel while the others update the trailing matrix; one barrier per step
- adaptive panel width (FORMULA2), so that the serial panel keeps up
- trailing update with shared packed U: needs the frame on a sub-team
  (threads 1..nt-1 of a region); with chunks of columns per thread every
  chunk packs its rows of L again
- fused laswp + trsm + pack (getrf_single): U packed once per panel
- threaded laswp (the interchanges to the left)
- panel teams (as the task variant) where the panel is the critical path
  (small n, tall matrices, many threads)

## Measurements

Drivers as for potrf: bgetrf.c (dlopen, ?getrf_ or ?getrf_64_, best of
3 runs after one warm up, uniform random matrix in [-1, 1), GFLOP/s (giga
floating-point operations per second) = m n^2 - n^3/3 for m >= n, x4
complex), cmpgetrf.c (pivots and factors against OpenBLAS, residual |P A -
L U| / (|A| k eps) with OpenBLAS's gemm).  ulmBLAS libulmblas_lbt.so built
with LAPACK=1 (?getrf_64_):

- current: 25013480, default build (getrf2 on the pool)
- tasks: 25013480 with OPENMP=1 (task graph; OMP_MAX_TASK_PRIORITY=3)
- proto: getrf-study (lookahead, below; default selection)

heim = E44 machines (Intel i5-12500, 6 cores, AVX2 (Advanced Vector
Extensions 2), GCC (GNU Compiler Collection) 12, -march=native), OpenBLAS
0.3.34 (pthreads, HASWELL kernels, ~/ulmbench/libopenblas.so); thread
count by ULMBLAS_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS.

## Prototype: lookahead on the pool (getrfla.h)

ulmblas/lapack/getrfla.{h,tcc}, getrfselect.h; getrf.tcc calls
getrfSelect.  One parallel region of the pool for the whole factorization,
no work space.  Step K (panel K factored):

1. thread 0: interchanges of panel K and the triangular solve in the next
   block column
2. all threads: the rows of the next block column (gemm, rows split)
3. thread 0: the next panel (getrf2, serial: nested BLAS calls inside the
   region run on one thread); the others start with 4
4. all threads: the columns right of the next block column in chunks
   (interchanges, trsm, gemm per chunk on one thread), guided: a chunk
   takes (remaining columns) / nt, at least 64, multiples of 8
5. barrier (counter)

At the end the interchanges left of each panel, panel by panel on all
threads.  Panel widths adaptive as OpenBLAS's FORMULA2: (n - s) (1 -
sqrt(1 - 1/nt)) between NBMin and NB (d: 64..256, z: 32..128), rounded to
8; every thread computes the widths from the position (no list of panels).
Threads: min(pool, min(m,n) / 64).

Variants for make tune: GetrfPlan (variant, nb, nbMin, chunk, threads);
built with -DULMBLAS_GETRF_STUDY the environment overrides them
(GETRF_VARIANT=r|l|t, GETRF_NB, GETRF_NBMIN, GETRF_CHUNK, GETRF_THREADS).

Steps on heim (d, 6 threads, n = 4000 unless noted):

- v1: thread 0 updated the whole next block column itself, chunks of
  ncols / (2 nt): 176 GFLOP/s; thread 0 spent 62 ms of 240 in that update
  (32 GFLOP/s, narrow gemm)
- v2: next block column split by rows over all threads: 200
- guided chunks instead of fixed ones: 208; adaptive widths: n = 1000
  63 -> 130 (fixed nb = 192 leaves 5 panels on the critical path), tall
  4000 x 1000 75 -> 161
- 2D tiles of the trailing update (each tile packs only its rows of L and
  columns of U): no gain (n = 8000 237 -> 201-237)
- chunk share: guide 1 (ncols/nt) 1-5 % better than ncols/(2 nt) and
  ncols/(3 nt) (syrlin, 3 rounds, median difference 0.9 %)

Profile (perf, d 8000, 6 threads): kernel 70 %, Steps (inlined swaps and
spinning) 10.5 %, packing of L 10 % (every chunk packs its rows of L
again), barrier waits about 5 % per worker.  OpenBLAS: kernel 66 %,
dlaswp_plus 14 %, blas_thread_server 11 %.  Independent gemms on column
chunks (m = n = 4000, k = 256, one chunk per thread, 6 threads): ulmBLAS
289, OpenBLAS 305 GFLOP/s; the threaded ulmBLAS gemm (shared packed B) 295;
k = 128: 227 (chunks) vs 265 (frame).  One thread: 59.6 vs 59.8.

## heim, 6 and 1 threads

proto = lookahead for every size (the selection before LookaheadMaxCols);
select = what getrfSelect now takes (lookahead up to 1000 columns per
thread, i.e. k <= 6000 on 6 threads, recursive above: the current
column); OB (OpenBLAS) in the ratios.  mauch, two rounds, all libraries
in the same run:

6 threads, GFLOP/s (best of rounds)

| t | m | n | current | tasks | proto | select | OpenBLAS | proto/current | select/OB | current/OB | tasks/OB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d | 1000 | 1000 | 92 | 153 | 135 | 135 | 172 | 1.47 | 0.79 | 0.54 | 0.89 |
| d | 2000 | 2000 | 146 | 196 | 173 | 173 | 226 | 1.18 | 0.76 | 0.65 | 0.87 |
| d | 4000 | 4000 | 208 | 213 | 210 | 210 | 242 | 1.01 | 0.87 | 0.86 | 0.88 |
| d | 8000 | 8000 | 258 | 195 | 237 | 258 | 277 | 0.92 | 0.93 | 0.93 | 0.71 |
| d | 16000 | 16000 | 310 | 210 | 269 | 310 | 293 | 0.87 | 1.06 | 1.06 | 0.72 |
| d | 4000 | 1000 | 132 | 194 | 163 | 163 | 175 | 1.24 | 0.93 | 0.75 | 1.11 |
| z | 1000 | 1000 | 147 | 189 | 189 | 189 | 265 | 1.28 | 0.71 | 0.56 | 0.71 |
| z | 2000 | 2000 | 212 | 200 | 225 | 225 | 295 | 1.06 | 0.76 | 0.72 | 0.68 |
| z | 4000 | 4000 | 264 | 174 | 263 | 263 | 294 | 1.00 | 0.89 | 0.90 | 0.59 |
| z | 8000 | 8000 | 318 | 194 | 288 | 318 | 311 | 0.91 | 1.02 | 1.02 | 0.62 |
| z | 4000 | 1000 | 196 | 170 | 187 | 187 | 270 | 0.95 | 0.69 | 0.73 | 0.63 |

round-to-round difference: median 1.2 %, 90th percentile 4.5 %

1 thread (harder; current, tasks and proto all run getrf2):

| t | m | n | current | OpenBLAS | current/OB |
|---|---:|---:|---:|---:|---:|
| d | 1000 | 1000 | 43 | 44 | 0.98 |
| d | 2000 | 2000 | 49 | 50 | 0.99 |
| d | 4000 | 4000 | 54 | 54 | 1.01 |
| d | 8000 | 8000 | 57 | 56 | 1.01 |
| d | 16000 | 16000 | 59 | 58 | 1.02 |
| d | 4000 | 1000 | 47 | 49 | 0.96 |
| z | 1000 | 1000 | 50 | 53 | 0.93 |
| z | 2000 | 2000 | 54 | 57 | 0.96 |
| z | 4000 | 4000 | 57 | 59 | 0.97 |
| z | 8000 | 8000 | 60 | 61 | 0.99 |
| z | 4000 | 1000 | 52 | 57 | 0.93 |

round-to-round difference: median 0.1 %

Panels on one thread (getrf2 vs OpenBLAS getrf_single, m x n): d 4000 x
128 29.0 vs 31.5, 4000 x 256 36.4 vs 38.1; z 4000 x 128 36.9 vs 45.1,
16000 x 128 33.9 vs 43.3.

Note: the first round of the first heim run (harder, 14:26) was disturbed
(current d 8000 156 instead of 258); the tables above are from clean runs.

- one thread: on par (d 0.96-1.02, z 0.93-0.99); the z panel is 0.8 of
  OpenBLAS's
- six threads: the lookahead helps where the recursion's small updates
  and serial swaps dominate (n <= 2000, tall: 1.2-1.5x current); from n =
  8000 on the recursive variant on the frame (shared packed blocks) is
  better, so the selection keeps it there
- still behind OpenBLAS at n = 1000-2000 (d 0.76-0.79, z 0.71-0.76) and z
  tall (0.69): the serial panel is the critical path (z panel 0.8 of
  OpenBLAS), and the trailing chunks pack their rows of L again (10 %) and
  run the kernel at about 53 GFLOP/s per thread in place of OpenBLAS's 60
- tasks (OPENMP=1) on 6 threads: good for d n <= 2000 and tall, bad for
  large n (0.71 of OpenBLAS at d 8000)

## Ice Lake, 64 and 32 threads

bwUniCluster cpu_il: Xeon Platinum 8358, 2 x 32 cores, SNC2 (sub-NUMA
clustering: 4 NUMA (non-uniform memory access) nodes), clang 19,
-march=native.  64 threads with numactl --interleave=all, 32 on socket 0
(--cpunodebind=0,1 --membind=0,1).  current, tasks, OpenBLAS
(benchmatrix/OpenBLAS-mt) and MKL (Intel Math Kernel Library) 2025.0
(MKL_THREADING_LAYER=GNU): job 8889424; proto (lookahead forced,
GETRF_VARIANT=l; proto16: on 16 threads) with OpenBLAS and MKL again: job
8892757 (OpenBLAS, MKL: best of both jobs).  Pivots and factors checked
against OpenBLAS (1001^2, 4001 x 1001, 1001 x 3001, 3001^2): pivots
equal, residual < 1.

64 threads, GFLOP/s (best of rounds)

| t | m | n | current | tasks | proto | proto16 | OpenBLAS | MKL | proto/current | proto/OB | tasks/OB | proto/MKL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d | 1000 | 1000 | 54 | 91 | 160 | 186 | 16 | 168 | 2.97 | 9.72 | 5.56 | 0.95 |
| d | 2000 | 2000 | 103 | 245 | 393 | 397 | 72 | 561 | 3.81 | 5.49 | 3.43 | 0.70 |
| d | 4000 | 4000 | 225 | 563 | 519 | 564 | 211 | 827 | 2.31 | 2.46 | 2.67 | 0.63 |
| d | 8000 | 8000 | 457 | 963 | 716 | 669 | 486 | 1752 | 1.57 | 1.47 | 1.98 | 0.41 |
| d | 16000 | 16000 | 915 | 1142 | 1050 | 846 | 1110 | 2465 | 1.15 | 0.95 | 1.03 | 0.43 |
| d | 4000 | 1000 | 94 | 239 | 207 | 213 | 53 | 41 | 2.20 | 3.94 | 4.54 | 5.08 |
| z | 1000 | 1000 | 109 | 318 | 332 | 304 | 66 | 370 | 3.04 | 5.05 | 4.83 | 0.90 |
| z | 2000 | 2000 | 202 | 673 | 542 | 631 | 193 | 808 | 2.68 | 2.81 | 3.49 | 0.67 |
| z | 4000 | 4000 | 431 | 1203 | 1072 | 861 | 447 | 1720 | 2.49 | 2.40 | 2.69 | 0.62 |
| z | 8000 | 8000 | 941 | 2082 | 1756 | 1060 | 1042 | 2650 | 1.87 | 1.68 | 2.00 | 0.66 |
| z | 4000 | 1000 | 193 | 474 | 372 | 410 | 118 | 75 | 1.93 | 3.15 | 4.02 | 4.93 |

32 threads, GFLOP/s (best of rounds)

| t | m | n | current | tasks | proto | proto16 | OpenBLAS | MKL | proto/current | proto/OB | tasks/OB | proto/MKL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d | 1000 | 1000 | 68 | 151 | 184 | 165 | 77 | 294 | 2.73 | 2.40 | 1.97 | 0.63 |
| d | 2000 | 2000 | 149 | 377 | 391 | 395 | 272 | 614 | 2.62 | 1.44 | 1.39 | 0.64 |
| d | 4000 | 4000 | 310 | 593 | 518 | 482 | 495 | 614 | 1.67 | 1.05 | 1.20 | 0.84 |
| d | 8000 | 8000 | 569 | 641 | 610 | 599 | 620 | 1279 | 1.07 | 0.98 | 1.03 | 0.48 |
| d | 16000 | 16000 | 910 | 681 | 846 | 751 | 885 | 1389 | 0.93 | 0.96 | 0.77 | 0.61 |
| d | 4000 | 1000 | 130 | 366 | 213 | 215 | 207 | 45 | 1.63 | 1.03 | 1.77 | 4.78 |
| z | 1000 | 1000 | 151 | 341 | 301 | 292 | 273 | 521 | 1.99 | 1.10 | 1.25 | 0.58 |
| z | 2000 | 2000 | 295 | 618 | 526 | 595 | 616 | 930 | 1.78 | 0.85 | 1.00 | 0.57 |
| z | 4000 | 4000 | 574 | 882 | 760 | 811 | 686 | 1531 | 1.32 | 1.11 | 1.29 | 0.50 |
| z | 8000 | 8000 | 976 | 1208 | 1109 | 988 | 1026 | 1707 | 1.14 | 1.08 | 1.18 | 0.65 |
| z | 4000 | 1000 | 280 | 508 | 339 | 378 | 448 | 73 | 1.21 | 0.76 | 1.13 | 4.63 |

round-to-round difference: median 0.9 %, 90th percentile 6.7 %

- the default build (current: getrf2, threads only inside BLAS) does not
  scale: 54-457 GFLOP/s for d n <= 8000 on 64 threads, 0.55-0.63 of
  OpenBLAS at n = 2000-4000 on 32 threads
- lookahead: 1.2-3.8x current on 64 threads, 1.1-2.7x on 32 threads (d
  16000 on 32 threads 0.93; the selection takes it there anyway: 16000 <=
  1000 x 32); against OpenBLAS 0.95-9.7 on 64, 0.76-2.4 on 32 threads (z
  2000 0.85, z tall 0.76)
- tasks (OPENMP=1) better than the lookahead from n = 4000 on 64 threads
  (d 8000 963 vs 716, z 8000 2082 vs 1756) and for z and tall on 32
  threads; the lookahead better for n <= 2000 on 64 threads
- proto16 (16 of 64 threads) about as fast up to n = 4000, slower above:
  no reason for a cap
- MKL 1.5-2.5x ahead from n = 2000 on; MKL tall 4000 x 1000: 41-75 GFLOP/s
  in both jobs (sic)

## Zen 4, 96 and 48 threads

bwUniCluster: AMD EPYC 9454, 2 x 48 cores, 8 NUMA nodes, clang 19,
-march=native; 96 threads interleaved, 48 on socket 0 (NUMA nodes 0-3).
current, tasks, OpenBLAS, MKL: job 8889425 (cpu); proto, proto16 with
OpenBLAS and MKL again: job 8896013 (dev_cpu, lscpu: EPYC 9454).  Pivots
and factors checked against OpenBLAS as on Ice Lake.

96 threads, GFLOP/s (best of rounds)

| t | m | n | current | tasks | proto | proto16 | OpenBLAS | MKL | proto/current | proto/OB | current/OB | tasks/OB | proto/MKL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d | 1000 | 1000 | 54 | 99 | 158 | 123 | 7 | 84 | 2.90 | 21.79 | 7.51 | 13.70 | 1.89 |
| d | 2000 | 2000 | 101 | 252 | 357 | 337 | 33 | 208 | 3.53 | 10.69 | 3.02 | 7.54 | 1.72 |
| d | 4000 | 4000 | 209 | 543 | 827 | 495 | 162 | 579 | 3.95 | 5.09 | 1.29 | 3.34 | 1.43 |
| d | 8000 | 8000 | 469 | 775 | 820 | 609 | 260 | 1036 | 1.75 | 3.15 | 1.80 | 2.98 | 0.79 |
| d | 16000 | 16000 | 1030 | 1027 | 1751 | 687 | 470 | 2220 | 1.70 | 3.72 | 2.19 | 2.19 | 0.79 |
| d | 4000 | 1000 | 93 | 179 | 172 | 193 | 35 | 46 | 1.85 | 4.86 | 2.62 | 5.03 | 3.74 |
| z | 1000 | 1000 | 102 | 185 | 185 | 184 | 30 | 256 | 1.81 | 6.18 | 3.42 | 6.19 | 0.72 |
| z | 2000 | 2000 | 207 | 443 | 306 | 324 | 139 | 497 | 1.48 | 2.20 | 1.49 | 3.19 | 0.61 |
| z | 4000 | 4000 | 458 | 985 | 579 | 489 | 351 | 961 | 1.26 | 1.65 | 1.30 | 2.80 | 0.60 |
| z | 8000 | 8000 | 1033 | 1501 | 936 | 631 | 596 | 1432 | 0.91 | 1.57 | 1.73 | 2.52 | 0.65 |
| z | 4000 | 1000 | 215 | 350 | 171 | 168 | 128 | 46 | 0.80 | 1.34 | 1.68 | 2.73 | 3.71 |

48 threads, GFLOP/s (best of rounds)

| t | m | n | current | tasks | proto | proto16 | OpenBLAS | MKL | proto/current | proto/OB | current/OB | tasks/OB | proto/MKL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d | 1000 | 1000 | 77 | 106 | 150 | 150 | 31 | 159 | 1.95 | 4.89 | 2.50 | 3.46 | 0.94 |
| d | 2000 | 2000 | 141 | 284 | 426 | 370 | 145 | 306 | 3.02 | 2.95 | 0.98 | 1.96 | 1.39 |
| d | 4000 | 4000 | 286 | 511 | 789 | 518 | 232 | 591 | 2.76 | 3.40 | 1.23 | 2.20 | 1.34 |
| d | 8000 | 8000 | 567 | 703 | 782 | 618 | 357 | 1053 | 1.38 | 2.19 | 1.59 | 1.97 | 0.74 |
| d | 16000 | 16000 | 995 | 816 | 1040 | 703 | 758 | 1223 | 1.04 | 1.37 | 1.31 | 1.08 | 0.85 |
| d | 4000 | 1000 | 131 | 220 | 249 | 240 | 120 | 49 | 1.89 | 2.07 | 1.10 | 1.83 | 5.10 |
| z | 1000 | 1000 | 145 | 206 | 224 | 226 | 115 | 339 | 1.54 | 1.95 | 1.27 | 1.80 | 0.66 |
| z | 2000 | 2000 | 293 | 526 | 432 | 385 | 343 | 599 | 1.47 | 1.26 | 0.85 | 1.54 | 0.72 |
| z | 4000 | 4000 | 565 | 954 | 671 | 523 | 495 | 905 | 1.19 | 1.36 | 1.14 | 1.93 | 0.74 |
| z | 8000 | 8000 | 1000 | 1317 | 853 | 647 | 949 | 1286 | 0.85 | 0.90 | 1.05 | 1.39 | 0.66 |
| z | 4000 | 1000 | 303 | 406 | 238 | 220 | 342 | 48 | 0.79 | 0.70 | 0.89 | 1.19 | 4.98 |

round-to-round difference: median 1.0 %, 90th percentile 5.8 %

- OpenBLAS is weak on Zen 4 (7-470 GFLOP/s on 96 threads); current is
  ahead of it except z 2000 and z tall on 48 threads (0.85-0.89)
- lookahead: 1.4-4x current for d (d 16000 on 96 threads 1751 vs 1030,
  0.79 of MKL), z up to 4000 1.2-1.8x; z 8000 (0.85-0.91) and z tall
  (0.79-0.80) better recursive, the selection takes the lookahead there
  (k <= 1000 threads): open for make tune
- tasks (OPENMP=1) best for z from 2000 on and for tall; lookahead best
  for d
- z tall 4000 x 1000 on 48 threads: lookahead 0.70 of OpenBLAS (current
  0.89, tasks 1.19)

## Selection (getrfselect.h, initial values)

- one thread or min(m,n) < 256: getrf2 (recursive), as before
- otherwise threads = min(pool, min(m,n)/64); lookahead while min(m,n) <=
  1000 x threads, above that getrf2 (heim: d and z from 8000 on)
- OPENMP=1 and 9 or more threads: the task graph as before; fewer
  threads: as the default build (heim: tasks 0.59-0.72 of OpenBLAS for z
  and d n >= 8000)

## Candidates (next)

- trailing update of the lookahead on the frame with a sub-team (threads
  1..nt-1 of the region share the packed U as the frame does): removes
  the repacking of L per chunk (10 % at d 8000 on heim), should make the
  lookahead the better variant also for large n (heim d 8000: 237 vs
  recursive 258, OpenBLAS 277)
- faster serial panel (critical path for n <= 2000 and tall): z getrf2 is
  0.8 of OpenBLAS's getrf_single (fused laswp + trsm + pack, Crout getf2);
  or a panel team (as the task variant) when the workers wait
- gemm selection for one thread inside a region (the 12x4 kernel at ~53
  GFLOP/s per thread in the chunks vs OpenBLAS's 4x8 at ~60)
- make tune: LookaheadMaxCols per type (z: recursive better from 8000 on
  heim and Zen 4 48 threads, lookahead better on Ice Lake), tasks vs
  lookahead in OPENMP=1 builds by threads and n, NB/NBMin
- int IndexType: frametrsm.h does not compile with IndexType = int
  (std::min of int and std::ptrdiff_t, lines 212, 481, 497, 499); getrf2
  and the lookahead with int indices are affected (the library uses
  std::ptrdiff_t)

## Tests

mauch (heim), GCC 12, make -j6: check-numeric PASS 1959, FAIL 0, XFAIL 9
(T0603 only), SKIP 436 (NEON (Advanced SIMD) and SME (Scalable Matrix
Extension) configurations on x86); new T0425 (getrfLookahead with fixed
and adaptive widths, 3, 4 and 7 threads, square, wide, tall, singular,
all strided layouts, s/d/c/z) passes in every configuration;
check-netlib PASS 207, FAIL 0; check-reference-lapack-lin (GE, PO, TR)
4 PASS, 0 FAIL (NMAX 132: serial path only).  Pivots and factors of the
lbt library against OpenBLAS (cmpgetrf) on heim with 2, 3, 4, 6 and 12
threads and on both cluster machines: pivots equal, residual < 1.
