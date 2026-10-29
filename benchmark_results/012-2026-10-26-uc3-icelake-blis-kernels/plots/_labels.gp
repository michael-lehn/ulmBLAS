#  Labels of campaign 012-2026-10-26-uc3-icelake-blis-kernels (loaded by the plot
#  scripts): variants (columns of data/summary/*.csv), titles, 1 = micro
#  kernel in ulmBLAS, 0 = native library; ratios to the native reference.
CPU = "Intel Xeon Platinum 8358 (Ice Lake)"
array V[16] = ["ulmblas_avx_k2", "ulmblas_haswell_6x8_k2", "ulmblas_haswell_8x6_k2", "ulmblas_sandybridge_8x4_k2", "ulmblas_penryn_4x4_k2", "ulmblas_skx_16x14_k2", "ulmblas_skx_16x12_l2_k2", "ulmblas_zen4_32x6_k2", "ulmblas_zen4_8x24_k2", "blis_auto", "blis_haswell", "blis_sandybridge", "blis_penryn", "openblas", "mkl", "aocl"]
array T[16] = ["ulmBLAS, own AVX/FMA 6x8", "ulmBLAS + BLIS haswell 6x8", "ulmBLAS + BLIS haswell 8x6", "ulmBLAS + BLIS sandybridge 8x4", "ulmBLAS + BLIS penryn 4x4", "ulmBLAS + BLIS skx 16x14", "ulmBLAS + BLIS skx 16x12 l2", "ulmBLAS + AOCL zen4 32x6", "ulmBLAS + AOCL zen4 8x24", "BLIS 2.0 skx (native)", "BLIS 2.0 haswell (native)", "BLIS 2.0 sandybridge (native)", "BLIS 2.0 penryn (native)", "OpenBLAS 0.3.34 SKYLAKEX", "oneMKL 2025.0.1", "AOCL-BLAS 5.0 (native, zen4)"]
array K[16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
array R[9] = ["ulmblas_avx_k2_over_ref", "ulmblas_haswell_6x8_k2_over_ref", "ulmblas_haswell_8x6_k2_over_ref", "ulmblas_sandybridge_8x4_k2_over_ref", "ulmblas_penryn_4x4_k2_over_ref", "ulmblas_skx_16x14_k2_over_ref", "ulmblas_skx_16x12_l2_k2_over_ref", "ulmblas_zen4_32x6_k2_over_ref", "ulmblas_zen4_8x24_k2_over_ref"]
array RT[9] = ["own AVX/FMA 6x8 / BLIS haswell", "haswell 6x8 / BLIS haswell", "haswell 8x6 / BLIS haswell", "sandybridge 8x4 / BLIS sandybridge", "penryn 4x4 / BLIS penryn", "skx 16x14 / BLIS skx", "skx 16x12 l2 / BLIS skx", "zen4 32x6 / AOCL (native)", "zen4 8x24 / AOCL (native)"]
PYMM = 54.20
PZMM = 105.19
YMAX = 112
