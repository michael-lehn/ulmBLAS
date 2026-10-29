#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
D = "../data/summary/"

set style line 1 lc rgb "#2a78d6" lw 3.0 pt 7 ps 0.7     # ulmBLAS + OpenBLAS kernel, iteration 1
set style line 2 lc rgb "#2a78d6" lw 1.6 pt 6 ps 0.7 dt (6,4)   # ... baseline
set style line 3 lc rgb "#3a3a38" lw 2.2 pt 4 ps 0.7     # native OpenBLAS
set style line 4 lc rgb "#e8a33d" lw 1.6 pt 9 ps 0.7     # ulmBLAS AVX/FMA
set style line 5 lc rgb "#1baf7a" lw 1.6 pt 13 ps 0.7    # native BLIS
set style line 9 lc rgb "#6b6a64" lw 1.2 dt 3            # reference lines

#  column C of the rows of variant V for square shapes
sq(v, c) = (strcol("variant") eq v && column("m")==column("n") && column("n")==column("k")) ? column(c) : NaN
