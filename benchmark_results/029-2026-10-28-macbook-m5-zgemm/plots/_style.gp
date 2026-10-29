#  Common style of the plots of this campaign (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').  ulmBLAS: solid lines; native libraries:
#  dashed lines.
set datafile separator ","
set datafile columnheaders
set grid xtics ytics lc rgb "#dcdcdc" lw 1 dt 1
set border lw 1.0 lc rgb "#52514e"
set tics nomirror out textcolor rgb "#0b0b0b"
D = "../data/summary/"
CPU = "Apple M5 (MacBook Air), one performance core"
array V[5] = ["ulmblas_sme", "ulmblas_neon", "accelerate", "eigen", "openblas"]
array T[5] = ["ulmBLAS, SME2 16x16 kernel", "ulmBLAS, NEON 4x3 kernel", \
              "Apple Accelerate", "Eigen 5.0.1", "OpenBLAS 0.3.34 (vortexm4)"]
array C[5] = ["#2a78d6", "#008300", "#eb6834", "#e87ba4", "#6250d6"]
array K[5] = [1, 1, 0, 0, 0]
array P[5] = [7, 9, 6, 8, 12]
