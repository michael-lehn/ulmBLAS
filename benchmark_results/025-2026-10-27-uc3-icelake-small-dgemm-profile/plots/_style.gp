#  Common style (loaded by the other scripts; not plotted by itself).
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
D = "../data/summary/"
set style line 1 lc rgb "#2a78d6" lw 2.6 pt 7 ps 0.7     # ulmBLAS
set style line 3 lc rgb "#3a3a38" lw 2.0 pt 4 ps 0.7     # OpenBLAS
set style line 9 lc rgb "#6b6a64" lw 1.2 dt 3            # reference lines
