#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#  Production configuration of the SME2 backend: MC, KC, NC = 384, 1024,
#  1024 (blue, thick); bands: minimum to maximum over the rounds; dotted
#  vertical lines: multiples of KC = NC = 1024.
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
D = "../data/summary/"
P = "b384_1024_1024"

set style line 1 lc rgb "#2a78d6" lw 3.0 pt 7 ps 0.6     # production
set style line 2 lc rgb "#e8a33d" lw 1.6 pt 7 ps 0.5
set style line 3 lc rgb "#c4553a" lw 1.6 pt 5 ps 0.5
set style line 4 lc rgb "#7d4fc4" lw 1.6 pt 9 ps 0.6
set style line 5 lc rgb "#1baf7a" lw 1.6 pt 13 ps 0.6
set style line 6 lc rgb "#8c8c8c" lw 1.6 pt 11 ps 0.6
set style line 7 lc rgb "#3a3a38" lw 2.0 dt (12,4,3,4) pt 4 ps 0.6    # BLIS
set style line 9 lc rgb "#6b6a64" lw 1.2 dt 3                          # reference lines

#  value of column C of a row of configuration V and the shape selection S
#  (S is an expression; rows not selected give NaN)
sel(v, c, s) = (strcol("config") eq v && s) ? column(c) : NaN

#  five sizes per panel of the coarse grid plots (fixed order of the colors)
array SC[5] = ["#2a78d6", "#e8a33d", "#c4553a", "#7d4fc4", "#1baf7a"]
array SP[5] = [7, 5, 9, 13, 11]
