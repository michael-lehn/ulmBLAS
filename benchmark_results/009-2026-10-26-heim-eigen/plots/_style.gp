#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#
#  ulmBLAS AVX/FMA current (blue, thick), BLIS and OpenBLAS as in the
#  campaigns 001-2026-10-24-heim-avx-fma and 008-2026-10-26-heim-avx-fma-memory,
#  Eigen (purple).  Bands: minimum to maximum over the rounds.
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
D = "../data/summary/"

set style line 4 lc rgb "#2a78d6" lw 3.2 pt 7 ps 0.6                       # ulmBLAS AVX current
set style line 7 lc rgb "#3a3a38" lw 2.0 dt (12,4,3,4) pt 4 ps 0.6         # BLIS
set style line 8 lc rgb "#1baf7a" lw 2.0 pt 13 ps 0.7                      # OpenBLAS
set style line 10 lc rgb "#9b3fb5" lw 2.6 pt 5 ps 0.6                      # Eigen

L4 = "ulmBLAS AVX/FMA (febb852)"
L7 = "BLIS 2.0 (haswell)"
L8 = "OpenBLAS 0.3.34 (HASWELL)"
L10 = "Eigen 5.0.1, C.noalias() += A*B"
