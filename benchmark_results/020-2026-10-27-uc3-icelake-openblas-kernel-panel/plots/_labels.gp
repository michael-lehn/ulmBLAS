#  Labels of campaign 020-2026-10-27-uc3-icelake-openblas-kernel-panel (loaded by
#  the plot scripts).
CPU = "Intel Xeon Platinum 8358 (Ice Lake)"
array V[5] = ["ulmblas_openblas_skylakex_16x12_k5", "ulmblas_openblas_skylakex_16x12_k3", "ulmblas_zen4_32x6_k2", "openblas", "mkl"]
array T[5] = ["ulmBLAS + OpenBLAS 16x12, one call per panel of B", "ulmBLAS + OpenBLAS 16x12, one call per tile", "ulmBLAS + AOCL zen4 32x6", "OpenBLAS 0.3.34 SKYLAKEX (native)", "oneMKL 2025.0.1"]
array K[5] = [1, 1, 1, 0, 0]
array R[3] = ["ulmblas_openblas_skylakex_16x12_k5_over_ref", "ulmblas_openblas_skylakex_16x12_k3_over_ref", "ulmblas_zen4_32x6_k2_over_ref"]
array RT[3] = ["OpenBLAS 16x12 per panel / OpenBLAS", "OpenBLAS 16x12 per tile / OpenBLAS", "zen4 32x6 / OpenBLAS"]
PYMM = 54.20
PZMM = 105.19
YMAX = 112
