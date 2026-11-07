# Row stride sweep (gnuplot >= 5.4).
#
#   gnuplot -e "R='getrf'; NAME='DGETRF'; D='results/<campaign>'; O='plots'" scripts/plot-sweep.gp
#
# D/summary-sweep-R.dat (scripts/summarize-sweep.py): column 1 R, then for
# inplace, colmajor, tiled and n = 500, 1000, 2000, 4000 pairs (gflops, rel);
# rel = time in place with R = 1 / time.
#   inplace:  rel in columns 3, 5, 7, 9
#   colmajor: rel in columns 11, 13, 15, 17
# Writes O/R-sweep.svg.
#
# Color = n (categorical slots 1-4), solid = in place, dashed = colmajor copy.

C1 = "#2a78d6"; C2 = "#eb6834"; C3 = "#1baf7a"; C4 = "#eda100"
INK = "#52514e"
if (!exists("SUB")) SUB = "Xeon Platinum 8358 Ice Lake, one socket, 32 threads"

set terminal svg size 900,520 dynamic enhanced font "Helvetica,13" background rgb "white"
set border 3 lw 1.0 lc rgb INK
set tics nomirror tc rgb INK
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key outside right top Left reverse box lc rgb "#dcdcdc" spacing 1.25 width 1
set logscale x 2
set xrange [0.9:140]
set xtics ("1" 1, "2" 2, "4" 4, "8" 8, "16" 16, "32" 32, "42" 42, "64" 64, "128" 128)
set xlabel "incRow (log scale)"
set yrange [0:1.3]
set ylabel "time with incRow = 1, in place / time"
F = D."/summary-sweep-".R.".dat"

set output O."/".R."-sweep.svg"
set title NAME.", ".SUB.": how much does the row stride cost?"
plot 1 w l lc rgb INK dt 2 lw 1.2 notitle, \
     F u 1:3  w lp lc rgb C1 lw 2 pt 7 ps 0.6 t "in place, n = 500", \
     F u 1:5  w lp lc rgb C2 lw 2 pt 7 ps 0.6 t "in place, n = 1000", \
     F u 1:7  w lp lc rgb C3 lw 2 pt 7 ps 0.6 t "in place, n = 2000", \
     F u 1:9  w lp lc rgb C4 lw 2 pt 7 ps 0.6 t "in place, n = 4000", \
     F u 1:11 w lp lc rgb C1 lw 2 dt 2 pt 6 ps 0.6 t "colmajor copy, n = 500", \
     F u 1:13 w lp lc rgb C2 lw 2 dt 2 pt 6 ps 0.6 t "colmajor copy, n = 1000", \
     F u 1:15 w lp lc rgb C3 lw 2 dt 2 pt 6 ps 0.6 t "colmajor copy, n = 2000", \
     F u 1:17 w lp lc rgb C4 lw 2 dt 2 pt 6 ps 0.6 t "colmajor copy, n = 4000"
