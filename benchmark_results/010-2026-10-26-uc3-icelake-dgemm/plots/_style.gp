#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#
#  ulmBLAS AVX/FMA (blue, thick), BLIS 2.0 auto-selected configuration
#  (dark grey, dashed) and haswell configuration (light grey, dotted),
#  OpenBLAS (green), oneMKL (orange), AOCL-BLAS (red), Eigen (purple).
#  Dashed horizontal lines: FMA peak of the core with 256-bit and, where
#  higher, 512-bit vectors (peak.c).  Bands: minimum to maximum over the
#  rounds.
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
D = "../data/summary/"

set style line 1 lc rgb "#2a78d6" lw 3.2 pt 7 ps 0.6                       # ulmBLAS
set style line 2 lc rgb "#3a3a38" lw 2.0 dt (12,4,3,4) pt 4 ps 0.6         # BLIS auto
set style line 3 lc rgb "#9a9a96" lw 2.0 dt (3,3) pt 6 ps 0.6              # BLIS haswell
set style line 4 lc rgb "#1baf7a" lw 2.0 pt 13 ps 0.7                      # OpenBLAS
set style line 5 lc rgb "#e8871e" lw 2.0 pt 9 ps 0.7                       # oneMKL
set style line 6 lc rgb "#c8323c" lw 2.0 dt (8,3) pt 11 ps 0.7             # AOCL-BLAS
set style line 7 lc rgb "#9b3fb5" lw 2.6 pt 5 ps 0.6                       # Eigen
set style line 9 lc rgb "#606060" lw 1.4 dt (6,4)                          # peaks
