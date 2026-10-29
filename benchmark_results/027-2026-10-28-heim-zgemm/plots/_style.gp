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
array T[8] = ["ulmBLAS + OpenBLAS HASWELL 4x2 kernel", "ulmBLAS + AOCL haswell 3x4 kernel", "ulmBLAS + BLIS haswell 3x4 kernel", "ulmBLAS + BLIS haswell 4x3 kernel", "ulmBLAS + BLIS sandybridge 4x4 kernel (AVX)", "ulmBLAS, reference kernel (round 1)", "OpenBLAS 0.3.34 (HASWELL)", "BLIS 2.0 (haswell)"]
array C[8] = ["#2a78d6", "#eda100", "#008300", "#e34948", "#6250d6", "#e87ba4", "#eb6834", "#1baf7a"]
array K[8] = [1, 1, 1, 1, 1, 1, 0, 0]
array P[8] = [7, 5, 9, 13, 11, 15, 3, 1]
