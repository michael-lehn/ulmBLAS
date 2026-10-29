#  Labels of campaign 013-2026-10-27-heim-blis-kernels-mc (loaded by the plot
#  scripts).
CPU = "Intel Core i5-12500"
array V[5] = ["ulmblas_avx", "ulmblas_haswell_6x8", "ulmblas_haswell_6x8_mc288", "ulmblas_haswell_8x6_mc288", "blis_haswell"]
array T[5] = ["ulmBLAS, own AVX/FMA 6x8 (MC 288)", "ulmBLAS + BLIS haswell 6x8, MC 72", "ulmBLAS + BLIS haswell 6x8, MC 288", "ulmBLAS + BLIS haswell 8x6, MC 288", "BLIS 2.0 haswell (native, MC 72)"]
array K[5] = [1, 1, 1, 1, 0]
array R[4] = ["ulmblas_avx_over_ref", "ulmblas_haswell_6x8_over_ref", "ulmblas_haswell_6x8_mc288_over_ref", "ulmblas_haswell_8x6_mc288_over_ref"]
array RT[4] = ["own AVX/FMA 6x8 / BLIS haswell", "haswell 6x8 MC 72 / BLIS haswell", "haswell 6x8 MC 288 / BLIS haswell", "haswell 8x6 MC 288 / BLIS haswell"]
PYMM = 0
PZMM = 0
YMAX = 66
