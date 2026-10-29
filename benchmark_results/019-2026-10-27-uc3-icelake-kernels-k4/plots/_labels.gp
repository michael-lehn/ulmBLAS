#  Labels of campaign 019-2026-10-27-uc3-icelake-kernels-k4 (loaded by the plot
#  scripts).
CPU = "Intel Xeon Platinum 8358 (Ice Lake)"
array V[8] = ["ulmblas_zen5_8x24_k4", "ulmblas_zen4_32x6_k2", "ulmblas_haswell_12x4_k4", "ulmblas_haswell_4x12_k4", "ulmblas_haswell_6x8_k2", "blis_haswell", "openblas", "aocl"]
array T[8] = ["ulmBLAS + AOCL 5.1 zen5 8x24", "ulmBLAS + AOCL zen4 32x6", "ulmBLAS + BLIS haswell 12x4 (old)", "ulmBLAS + BLIS haswell 4x12 (old)", "ulmBLAS + BLIS haswell 6x8", "BLIS 2.0 haswell (native)", "OpenBLAS 0.3.34 SKYLAKEX", "AOCL-BLAS 5.0 (native)"]
array K[8] = [1, 1, 1, 1, 1, 0, 0, 0]
array R[5] = ["ulmblas_zen5_8x24_k4_over_ref", "ulmblas_zen4_32x6_k2_over_ref", "ulmblas_haswell_12x4_k4_over_ref", "ulmblas_haswell_4x12_k4_over_ref", "ulmblas_haswell_6x8_k2_over_ref"]
array RT[5] = ["zen5 8x24 / OpenBLAS", "zen4 32x6 / OpenBLAS", "haswell 12x4 (old) / BLIS haswell", "haswell 4x12 (old) / BLIS haswell", "haswell 6x8 / BLIS haswell"]
PYMM = 54.20
PZMM = 105.19
YMAX = 112
