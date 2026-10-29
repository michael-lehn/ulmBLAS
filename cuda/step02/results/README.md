# Results of step 02

## 2026-10-29: steps 01 and 02, cuBLAS and MAGMA on an NVIDIA A100

[2026-10-29-a100/](2026-10-29-a100/): bwUniCluster 3.0, partition
`dev_gpu_a100_il`, NVIDIA A100 80GB PCIe, driver 615.71.09, CUDA 12.8
(cuBLAS 12.8.3), GCC 11.5, MAGMA 2.10.0 (built as described in
[../../step01/results/](../../step01/results/README.md)), Nsight Compute
2025.1; `make CUDA_ARCH=sm_80 MAGMA=...`, job script `cuda-step02.sbatch`
(`env.txt`: node, GPU, versions).

- `check.log`: 2310 checks of cULMblas, 0 failed (769 cases for step01,
  step02 and cULMblas::gemm); MAGMA as in step 01 (magmablas_dgemm differs
  from BLAS in 5 cases).  `memcheck.log`: 0 errors.
- `bench-square.log`: all implementations, m = n = k = 100, ..., 4000, median
  of 10 calls after 2 warm-up calls, GPU time between CUDA events.
- `bench-cols2.log`, `bench-cols8.log`: step 02 with 2 and 8 columns per
  thread (builds `build-cols2.log`, `build-cols8.log`).
- `ncu-step01.log`, `ncu-step02.log`: Nsight Compute metrics of one call
  at n = 2048 (load and store requests and sectors, hit rates, throughput
  of L1 and DRAM, activity of the FP64 pipe and of the load/store units,
  active warps, registers).

| n = m = k | cuBLAS | magmablas_dgemm | step 01 | step 02 | step 02 / step 01 |
|---:|---:|---:|---:|---:|---:|
| 500 | 7629 | 2570 | 1521 | 2230 | 1.47 |
| 1000 | 13951 | 4995 | 1917 | 2933 | 1.53 |
| 2000 | 16938 | 6871 | 1861 | 2820 | 1.52 |
| 3000 | 18327 | 6807 | 1842 | 2878 | 1.56 |
| 4000 | 16463 | 6896 | 1867 | 3003 | 1.61 |

(GFLOP/s.)
