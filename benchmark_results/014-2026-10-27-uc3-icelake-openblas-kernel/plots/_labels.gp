#  Labels of campaign 014-2026-10-27-uc3-icelake-openblas-kernel (loaded by the
#  plot scripts).
CPU = "Intel Xeon Platinum 8358 (Ice Lake)"
array V[7] = ["ulmblas_avx_k2", "ulmblas_skx_16x14_k2", "ulmblas_zen4_32x6_k2", "ulmblas_openblas_skylakex_16x12_k3", "blis_auto", "openblas", "mkl"]
array T[7] = ["ulmBLAS, own AVX/FMA 6x8", "ulmBLAS + BLIS skx 16x14", "ulmBLAS + AOCL zen4 32x6", "ulmBLAS + OpenBLAS SKYLAKEX 16x12", "BLIS 2.0 skx (native)", "OpenBLAS 0.3.34 SKYLAKEX (native)", "oneMKL 2025.0.1"]
array K[7] = [1, 1, 1, 1, 0, 0, 0]
array R[3] = ["ulmblas_openblas_skylakex_16x12_k3_over_ref", "ulmblas_zen4_32x6_k2_over_ref", "ulmblas_skx_16x14_k2_over_ref"]
array RT[3] = ["OpenBLAS 16x12 in ulmBLAS / OpenBLAS", "AOCL zen4 32x6 in ulmBLAS / OpenBLAS", "BLIS skx 16x14 in ulmBLAS / BLIS skx"]
PYMM = 54.20
PZMM = 105.19
YMAX = 112
