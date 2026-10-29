#  Common style of the plots of this campaign (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').  Every variant keeps its color in all
#  plots.  ulmBLAS (the first 7): solid lines; native libraries: dashed
set datafile separator ","
set datafile columnheaders
set datafile missing "nan"
set grid xtics ytics lc rgb "#dcdcdc" lw 1 dt 1
set border lw 1.0 lc rgb "#52514e"
set tics nomirror out textcolor rgb "#0b0b0b"
D = "../data/summary/"
CPU = "Intel Xeon Platinum 8358 (Ice Lake), one core"
array V[12] = ["ulmblas_openblas_skylakex_8x2", "ulmblas_openblas_haswell_8x2", "ulmblas_aocl_haswell_3x8", "ulmblas_haswell_3x8", "ulmblas_haswell_8x3", "ulmblas_sandybridge_8x4", "ulmblas_ref", "mkl", "openblas", "blis_auto", "aocl", "blis_haswell"]
array T[12] = ["ulmBLAS + OpenBLAS SKYLAKEX 8x2 kernel", "ulmBLAS + OpenBLAS HASWELL 8x2 kernel", "ulmBLAS + AOCL haswell 3x8 kernel", "ulmBLAS + BLIS haswell 3x8 kernel", "ulmBLAS + BLIS haswell 8x3 kernel", "ulmBLAS + BLIS sandybridge 8x4 kernel", "ulmBLAS, reference kernel (round 1)", "oneMKL 2025.0.1", "OpenBLAS 0.3.34 (SKYLAKEX)", "BLIS 2.0 (skx, 1m)", "AOCL-BLAS 5.0 (haswell 3x8)", "BLIS 2.0 (haswell)"]
array C[12] = ["#2a78d6", "#eda100", "#6250d6", "#e34948", "#e87ba4", "#7f7f00", "#a0a0a0", "#eb6834", "#8a8984", "#52514e", "#1baf7a", "#b07aa1"]
array P[12] = [7, 9, 11, 13, 15, 14, 1, 3, 6, 4, 8, 2]
