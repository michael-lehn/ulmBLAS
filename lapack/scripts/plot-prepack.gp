# Packed tiles (param prepack) vs. packing in every GEMM (gnuplot >= 5.4).
#
#   gnuplot -e "R='potrf'; NAME='DPOTRF'; D='results/<campaign>'; O='plots'" scripts/plot-prepack.gp
#
# D/summary-R-prepack.dat (scripts/summarize-strides.py): column 1 n, then
# (t, gflops) for 12 groups, then t_without / t_with prepack in columns
# 26-28 (incRow 1: inplace, colmajor, tiled) and 29-31 (incRow 42).
# Writes O/R-prepack.svg.

C1 = "#2a78d6"; C3 = "#1baf7a"; C4 = "#eda100"
INK = "#52514e"
if (!exists("SUB")) SUB = "Xeon Platinum 8358 Ice Lake, one socket, 32 threads"

set terminal svg size 900,520 dynamic enhanced font "Helvetica,13" background rgb "white"
set border 3 lw 1.0 lc rgb INK
set tics nomirror tc rgb INK
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key top right Left reverse box lc rgb "#dcdcdc" opaque spacing 1.25 width 1
set xlabel "m = n"
set xrange [0:4050]
set xtics 0,500,4000
set yrange [0.6:1.5]
set ylabel "time without / time with packed tiles"
F = D."/summary-".R."-prepack.dat"

set output O."/".R."-prepack.svg"
set title NAME.", ".SUB.": tiles packed once (prepack)"
plot 1 w l lc rgb INK dt 2 lw 1.2 notitle, \
     F u 1:26 w lp lc rgb C1 lw 2 pt 7 ps 0.5 t "in place, incRow = 1", \
     F u 1:27 w lp lc rgb C3 lw 2 pt 7 ps 0.5 t "colmajor copy, incRow = 1", \
     F u 1:28 w lp lc rgb C4 lw 2 pt 7 ps 0.5 t "tiled copy, incRow = 1", \
     F u 1:29 w lp lc rgb C1 lw 2 dt 2 pt 6 ps 0.5 t "in place, incRow = 42", \
     F u 1:30 w lp lc rgb C3 lw 2 dt 2 pt 6 ps 0.5 t "colmajor copy, incRow = 42", \
     F u 1:31 w lp lc rgb C4 lw 2 dt 2 pt 6 ps 0.5 t "tiled copy, incRow = 42"
