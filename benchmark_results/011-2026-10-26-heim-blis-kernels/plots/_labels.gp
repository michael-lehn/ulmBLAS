#  Labels of campaign 011-2026-10-26-heim-blis-kernels (loaded by the plot
#  scripts): variants (columns of data/summary/*.csv), titles, 1 = micro
#  kernel in ulmBLAS, 0 = native library; ratios to the native reference.
CPU = "Intel Core i5-12500"
array V[10] = ["ulmblas_avx", "ulmblas_haswell_6x8", "ulmblas_haswell_8x6", "ulmblas_sandybridge_8x4", "ulmblas_penryn_4x4", "ulmblas_sse", "blis_haswell", "blis_sandybridge", "blis_penryn", "openblas"]
array T[10] = ["ulmBLAS, own AVX/FMA 6x8 (port of BLIS haswell)", "ulmBLAS + BLIS haswell 6x8", "ulmBLAS + BLIS haswell 8x6", "ulmBLAS + BLIS sandybridge 8x4 (AVX)", "ulmBLAS + BLIS penryn 4x4 (SSE3)", "ulmBLAS, own SSE 4x4", "BLIS 2.0 haswell (native)", "BLIS 2.0 sandybridge (native)", "BLIS 2.0 penryn (native)", "OpenBLAS 0.3.34 HASWELL (native)"]
array K[10] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
array R[6] = ["ulmblas_avx_over_ref", "ulmblas_haswell_6x8_over_ref", "ulmblas_haswell_8x6_over_ref", "ulmblas_sandybridge_8x4_over_ref", "ulmblas_penryn_4x4_over_ref", "ulmblas_sse_over_ref"]
array RT[6] = ["own AVX/FMA 6x8 / BLIS haswell", "BLIS haswell 6x8 in ulmBLAS / BLIS haswell", "BLIS haswell 8x6 in ulmBLAS / BLIS haswell", "BLIS sandybridge 8x4 in ulmBLAS / BLIS sandybridge", "BLIS penryn 4x4 in ulmBLAS / BLIS penryn", "own SSE 4x4 / BLIS penryn"]
PYMM = 0
PZMM = 0
YMAX = 66
