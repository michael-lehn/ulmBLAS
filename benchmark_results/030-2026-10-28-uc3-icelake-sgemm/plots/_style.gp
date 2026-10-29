#  Common style of the plots of this campaign (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').  Every variant keeps its color in all
#  plots.  ulmBLAS (the first 11): solid lines; native libraries: dashed
set datafile separator ","
set datafile columnheaders
set datafile missing "nan"
set grid xtics ytics lc rgb "#dcdcdc" lw 1 dt 1
set border lw 1.0 lc rgb "#52514e"
set tics nomirror out textcolor rgb "#0b0b0b"
D = "../data/summary/"
CPU = "Intel Xeon Platinum 8358 (Ice Lake), one core"
array V[16] = ["ulmblas_openblas_skylakex_16x4", "ulmblas_skx_32x12_l2", "ulmblas_openblas_haswell_8x4", "ulmblas_haswell_6x16", "ulmblas_aocl_haswell_6x16", "ulmblas_haswell_16x6", "ulmblas_haswell_4x24", "ulmblas_haswell_24x4", "ulmblas_sandybridge_8x8", "ulmblas_penryn_8x4", "ulmblas_ref", "mkl", "openblas", "blis_auto", "aocl", "blis_haswell"]
array T[16] = ["ulmBLAS + OpenBLAS SKYLAKEX 16x4 kernel", "ulmBLAS + BLIS skx 32x12 l2 kernel", "ulmBLAS + OpenBLAS HASWELL 8x4 kernel", "ulmBLAS + BLIS haswell 6x16 kernel", "ulmBLAS + AOCL haswell 6x16 kernel", "ulmBLAS + BLIS haswell 16x6 kernel", "ulmBLAS + BLIS haswell 4x24 kernel", "ulmBLAS + BLIS haswell 24x4 kernel", "ulmBLAS + BLIS sandybridge 8x8 kernel", "ulmBLAS + BLIS penryn 8x4 kernel", "ulmBLAS, reference kernel (round 1)", "oneMKL 2025.0.1", "OpenBLAS 0.3.34 (SKYLAKEX)", "BLIS 2.0 (skx)", "AOCL-BLAS 5.0 (skx 32x12 l2)", "BLIS 2.0 (haswell)"]
array C[16] = ["#2a78d6", "#008300", "#eda100", "#e34948", "#6250d6", "#e87ba4", "#1f9fb4", "#9c6b30", "#7f7f00", "#b00068", "#a0a0a0", "#eb6834", "#8a8984", "#52514e", "#1baf7a", "#b07aa1"]
array P[16] = [7, 5, 9, 13, 11, 15, 2, 12, 14, 10, 1, 3, 6, 4, 8, 1]
