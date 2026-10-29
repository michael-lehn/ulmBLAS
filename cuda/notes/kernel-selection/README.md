# Note: choosing the kernel by more than m*n?

Not a step of the tutorial: an experiment after step 14 whose answer is
"no".  The selection of step 12 stays.

## The question

- Since step 12, `gemm` chooses one of three kernels by the size of C:
  - m*n < 1000^2: small (32 x 16 elements of C per block);
  - m*n < 3000^2: medium (64 x 32);
  - otherwise: large (64 x 64).
- The thresholds came from square matrices.  Does the best kernel also
  depend on k, on the shape of C, or on the number of blocks?

## The experiment

- Step 14 with one kernel forced for all shapes
  ([force-kernel.patch](results/2026-10-29-a100/force-kernel.patch)),
  and with the selection as it is.
- 64 shapes ([shapes.txt](results/2026-10-29-a100/shapes.txt)): 16 shapes
  of C, square (512 to 4096), tall and wide (16384 x 256 to 256 x 16384),
  each with k = 64, 256, 1024, 4096.
- 3 jobs, each kernel in its own build, in 3 different orders
  ([d1/, d2/, d3/](results/2026-10-29-a100/)); the tables use the median
  of the 3 runs.  Between the runs the same kernel and shape differ by 0
  to 2% for most shapes, up to 11% for a few.
- Full table: [table.txt](results/2026-10-29-a100/table.txt) (by
  [an.py](results/2026-10-29-a100/an.py), with the waves of each kernel).

## What the measurements show

| m*n | k | best kernel | compared with the selection |
|---|---|---|---|
| < 1000^2 (512 x 512) | all | small | as chosen |
| 1000^2 to 2000^2 | 256 to 4096 | medium | as chosen (1500^2 x 256: large 4% faster) |
| | 64 | large in 3 of 4 | large 12 to 27% faster, but calls of ~30 us |
| 2048^2 (2048 x 2048, 4096 x 1024, 16384 x 256, ...) | 256 | large | large 3.5 to 7% faster (all 7 shapes) |
| | 1024 | about equal | -0.2 to +1.2% |
| | 4096 | mixed | -4 to +2% |
| | 64 | small or medium | large 1 to 7% slower |
| >= 3000^2 | 256 to 4096 | large | as chosen |
| | 64 | small or medium | 1 to 4% faster than large |

- The shape of C (tall, wide, square) makes no visible difference beyond
  m*n: the 7 shapes with m*n = 2048^2 behave alike.
- k matters only near the threshold between medium and large, and not in
  one direction: small k favors the large kernel at k = 256, but not at
  k = 64.

## Rules tried

- Each rule scored with the measured times of the forced kernels, ratio
  to the selection of step 12 ([rules.py](results/2026-10-29-a100/rules.py),
  [rules.txt](results/2026-10-29-a100/rules.txt)):

| rule | geometric mean over the 64 shapes | worst shape | best shape |
|---|---:|---:|---:|
| m*n, as in step 12 | 1 | 1 | 1 |
| large from m*n >= 2048^2 | 1.0003 | 0.93 | 1.07 |
| large from m*n >= 2048^2 if k <= 1024 | 1.0008 | 0.93 | 1.07 |
| the best of the three for every shape (no rule can do better) | 1.018 | 1 | 1.27 |

- Even the perfect choice gains 1.8%, mostly for k = 64.  The simple
  rules gain less than 0.1% and lose up to 7% for single shapes: inside
  the noise of the measurements.
- So the selection stays as in step 12.  A better selection would need
  many special cases, or measurements at run time (an autotuner); not a
  small step.

## What this says about the kernels

- Between the medium and the large kernel there is little to choose for
  most shapes: the large kernel needs fewer copies per `mma.sync`, the
  medium one has twice as many blocks; both effects are small at these
  sizes.
- k = 64 is slow with every kernel (4 to 7 TFLOP/s, cuBLAS 4 to 8): the
  prologue (the first panels) and the epilogue (C) are a large part of a
  short calculation.
