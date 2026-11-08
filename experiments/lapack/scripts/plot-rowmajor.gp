# Row major vs. column major (gnuplot >= 5.4).
#
#   gnuplot -e "R='getrf'; NAME='DGETRF'; D='results/<campaign>'; O='plots'" scripts/plot-rowmajor.gp
#
# D/summary-R-rowmajor.dat (scripts/summarize-rowmajor.py): column 1 n, then
# (t, gflops, speedup over OpenBLAS column major) for openblas:colmajor (2-4),
# openblas:rowmajor (5-7), ulmblas:rowmajor (8-10), ulmblas:colmajor (11-13).
# Writes O/R-rowmajor-gflops.svg and O/R-rowmajor-speedup.svg.

C1 = "#2a78d6"; C2 = "#eb6834"; C3 = "#1baf7a"
INK = "#52514e"
if (!exists("SUB")) SUB = "Xeon Platinum 8358 Ice Lake, one socket, 32 threads"
F = D."/summary-".R."-rowmajor.dat"

set terminal svg size 900,520 dynamic enhanced font "Helvetica,13" background rgb "white"
set border 3 lw 1.0 lc rgb INK
set tics nomirror tc rgb INK
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key outside right top Left reverse box lc rgb "#dcdcdc" spacing 1.25 width 1
set xlabel "m = n"
set xrange [0:4050]
set xtics 0,500,4000

set output O."/".R."-rowmajor-gflops.svg"
set title NAME.", ".SUB.": row major"
set ylabel "GFLOP/s"
set yrange [0:*]
plot F u 1:9  w lp lc rgb C1 lw 2 pt 7 ps 0.4 t "ulmBLAS, row major", \
     F u 1:3  w lp lc rgb C2 lw 2 pt 7 ps 0.4 t "OpenBLAS, column major", \
     F u 1:6  w lp lc rgb C3 lw 2 pt 7 ps 0.4 t "OpenBLAS (LAPACKE), row major", \
     F u 1:12 w l  lc rgb C1 lw 1.5 dt 2 t "ulmBLAS, column major"

set output O."/".R."-rowmajor-speedup.svg"
set title NAME.", ".SUB.": speedup over OpenBLAS, column major"
set ylabel "time OpenBLAS column major / time"
set yrange [0:*]
plot 1 w l lc rgb INK dt 2 lw 1.2 notitle, \
     F u 1:10 w lp lc rgb C1 lw 2 pt 7 ps 0.4 t "ulmBLAS, row major", \
     F u 1:7  w lp lc rgb C3 lw 2 pt 7 ps 0.4 t "OpenBLAS (LAPACKE), row major", \
     F u 1:13 w l  lc rgb C1 lw 1.5 dt 2 t "ulmBLAS, column major"
