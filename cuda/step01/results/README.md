# Results of step 01

## 2026-10-29: step 01, cuBLAS and MAGMA on an NVIDIA A100

[2026-10-29-a100/](2026-10-29-a100/):

- Same node type and software as below (A100 80GB PCIe, driver 615.71.09,
  CUDA 12.8, cuBLAS 12.8.3, GCC 11.5).
- New: MAGMA 2.10.0, built from the release tarball (SHA-256 ea0c57fc…,
  `magma-build.sbatch`, `magma-build.txt`: `make.inc.mkl-gcc` with
  `GPU_TARGET=sm_80`, MKL 2025.0).
- `make CUDA_ARCH=sm_80 MAGMA=...`, job script `cuda-gpu-magma.sbatch`.
- `make check`: 1540 checks of cULMblas, 0 failed.  MAGMA runs through the
  same 769 cases: `magma_dgemm` (which calls `cublasDgemm`) differs from
  BLAS in none, `magmablas_dgemm` in 5 (`check.log`, lines `DIFF`): for
  alpha = 0 it reads A and B (NaN in A and B gives NaN in C), and for k = 0
  it returns without scaling C by beta.
- `make check-memcheck`: 0 errors, also for both MAGMA functions.
- `bench-square.log`: m = n = k = 100, 200, ..., 4000 (the plot of
  [../README.md](../README.md)); `bench-rect.log`: rectangular shapes;
  `bench-pad.log`: padded leading dimensions.  Median of 10 calls after 2
  warm-up calls, GPU time between CUDA events.

| n = m = k | cuBLAS | MAGMA magma_dgemm | MAGMA magmablas_dgemm | step 01 |
|---:|---:|---:|---:|---:|
| 500 | 7629 | 7629 | 2570 | 1526 |
| 1000 | 13951 | 13951 | 4995 | 1917 |
| 2000 | 17002 | 16984 | 6597 | 1861 |
| 3000 | 17855 | 17540 | 6810 | 1846 |
| 4000 | 16479 | 16124 | 6927 | 1869 |

(GFLOP/s.)

## 2026-10-28: step 01 (naive) on an NVIDIA A100

[2026-10-28-a100/](2026-10-28-a100/):

- bwUniCluster 3.0, partition `dev_gpu_a100_il`, one NVIDIA A100 80GB PCIe
  (compute capability 8.0, 108 SMs).
- Driver 615.71.09, CUDA 12.8 (nvcc 12.8.61, cuBLAS 12.8.3), host compiler
  GCC 11.5, `make CUDA_ARCH=sm_80`.
- Job script `cuda-gpu.sbatch` (`env.txt`: node, GPU, versions).
- The kernel was called `naive` in these logs (now `step01`).
- `make check`: 1540 checks, 0 failed (769 cases for `naive` and for
  `cULMblas::gemm`, and the argument checks; `check.log`).
- `make check-memcheck`: the quick cases under compute-sanitizer
  (memcheck), 0 errors (`memcheck.log`).
- `make bench` (`bench.log`, `bench-pad.log` with padded leading
  dimensions): DGEMM, alpha = beta = 1, median of 10 calls after 2 warm-up
  calls, GPU time between CUDA events:

| m x n x k | cuBLAS [GFLOP/s] | naive [GFLOP/s] | naive / cuBLAS |
|---|---:|---:|---:|
| 256³ | 2521 | 936 | 0.371 |
| 1024³ | 15768 | 2030 | 0.129 |
| 2048³ | 17754 | 1974 | 0.111 |
| 4096³ | 16773 | 1920 | 0.114 |
| 4096 x 4096 x 256 | 16578 | 2159 | 0.130 |
| 4096 x 256 x 4096 | 17623 | 1797 | 0.102 |
| 256 x 4096 x 4096 | 17225 | 1872 | 0.109 |

- The naive kernel: about 2 TFLOP/s, a ninth of cuBLAS for the larger
  sizes; the baseline for the next steps.
- Small sizes: the ratio is higher because cuBLAS does not reach its speed
  there either.
