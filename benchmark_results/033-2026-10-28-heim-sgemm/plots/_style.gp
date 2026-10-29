#  Common style of the plots of this campaign (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').  Every variant keeps its color in all
#  plots.  ulmBLAS: solid lines; native libraries: dashed lines.
set datafile separator ","
set datafile columnheaders
set datafile missing "nan"
set grid xtics ytics lc rgb "#dcdcdc" lw 1 dt 1
set border lw 1.0 lc rgb "#52514e"
set tics nomirror out textcolor rgb "#0b0b0b"
D = "../data/summary/"
CPU = "heim: Intel Core i5-12500 (Alder Lake), core 3"
array V[8] = ["ulmblas_openblas_haswell_4x2", "ulmblas_aocl_haswell_3x4", "ulmblas_haswell_3x4", "ulmblas_haswell_4x3", "ulmblas_sandybridge_4x4", "ulmblas_ref", "openblas", "blis"]
array V[11] = ["ulmblas_openblas_haswell_8x4", "ulmblas_aocl_haswell_6x16", "ulmblas_haswell_6x16", "ulmblas_haswell_16x6", "ulmblas_haswell_4x24", "ulmblas_haswell_24x4", "ulmblas_sandybridge_8x8", "ulmblas_penryn_8x4", "ulmblas_ref", "openblas", "blis"]
array T[11] = ["ulmBLAS + OpenBLAS HASWELL 8x4 kernel", "ulmBLAS + AOCL haswell 6x16 kernel", "ulmBLAS + BLIS haswell 6x16 kernel", "ulmBLAS + BLIS haswell 16x6 kernel", "ulmBLAS + BLIS haswell 4x24 kernel", "ulmBLAS + BLIS haswell 24x4 kernel", "ulmBLAS + BLIS sandybridge 8x8 kernel (AVX)", "ulmBLAS + BLIS penryn 8x4 kernel (SSE3)", "ulmBLAS, reference kernel (round 1)", "OpenBLAS 0.3.34 (HASWELL)", "BLIS 2.0 (haswell)"]
array C[11] = ["#2a78d6", "#eda100", "#008300", "#e34948", "#8a5a00", "#52514e", "#6250d6", "#00a3c4", "#e87ba4", "#eb6834", "#1baf7a"]
array K[11] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
array P[11] = [7, 5, 9, 13, 11, 15, 4, 6, 2, 3, 1]
