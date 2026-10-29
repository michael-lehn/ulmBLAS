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
CPU = "Intel Xeon Platinum 8358 (Ice Lake), one core"
array V[11] = ["ulmblas_zen4_12x4", "ulmblas_zen4_4x12", "ulmblas_openblas_skylakex_4x2", "ulmblas_openblas_haswell_4x2", "ulmblas_haswell_3x4", "ulmblas_ref", "mkl", "aocl", "openblas", "blis_auto", "blis_haswell"]
array T[11] = ["ulmBLAS + AOCL zen4 12x4 kernel", "ulmBLAS + AOCL zen4 4x12 kernel", "ulmBLAS + OpenBLAS SKYLAKEX 4x2 kernel", "ulmBLAS + OpenBLAS HASWELL 4x2 kernel", "ulmBLAS + BLIS haswell 3x4 kernel", "ulmBLAS, reference kernel (round 1)", "oneMKL 2025.0.1", "AOCL-BLAS 5.0 (zen4 12x4)", "OpenBLAS 0.3.34 (SKYLAKEX)", "BLIS 2.0 (skx, 1m)", "BLIS 2.0 (haswell)"]
array C[11] = ["#2a78d6", "#eda100", "#008300", "#e34948", "#6250d6", "#e87ba4", "#eb6834", "#1baf7a", "#8a8984", "#52514e", "#b07aa1"]
array K[11] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
array P[11] = [7, 5, 9, 13, 11, 15, 3, 1, 6, 4, 8]
