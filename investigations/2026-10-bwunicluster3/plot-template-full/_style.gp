#  Common style of the plots of the full DGEMM campaign (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').  Every variant keeps its color in all plots
#  (categorical palette of 8 hues, validated for color vision deficiencies
#  on the white background; AOCL-BLAS, which does not target this CPU, in
#  gray).  ulmBLAS: solid lines and filled markers; native libraries:
#  dashed lines and open markers.
set datafile separator ","
set datafile columnheaders
set grid xtics ytics lc rgb "#dcdcdc" lw 1 dt 1
set border lw 1.0 lc rgb "#52514e"
set tics nomirror out textcolor rgb "#0b0b0b"
D = "../data/summary/"
CPU = "Intel Xeon Platinum 8358 (Ice Lake), one core"
PZMM = 105.19          # FMA peak with zmm (512 bit), data/raw/peak-before.txt
PYMM = 54.20           # FMA peak with ymm (256 bit)

#  the 9 variants of the GFLOP/s plots: column, title, color, ulmBLAS?
array V[9] = ["ulmblas_openblas_skylakex_16x12_it3", "ulmblas_zen4_32x6_f1", \
              "ulmblas_skx_16x14_f1", "ulmblas_avx_f1", "openblas", "mkl", \
              "eigen", "blis_auto", "aocl"]
array T[9] = ["ulmBLAS + OpenBLAS 16x12 kernel (final)", \
              "ulmBLAS + AOCL zen4 32x6 kernel", \
              "ulmBLAS + BLIS skx 16x14 kernel", \
              "ulmBLAS, own AVX2/FMA 6x8 kernel", \
              "OpenBLAS 0.3.34 (SKYLAKEX)", "oneMKL 2025.0.1", \
              "Eigen 5.0.1", "BLIS 2.0 (skx)", "AOCL-BLAS 5.0"]
array C[9] = ["#2a78d6", "#eda100", "#008300", "#e34948", \
              "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#8a8984"]
array K[9] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
array P[9] = [7, 5, 9, 13, 6, 4, 8, 12, 10]
