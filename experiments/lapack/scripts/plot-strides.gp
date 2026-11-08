# Plots of a row stride campaign (gnuplot >= 5.4).
#
#   gnuplot -e "R='getrf'; NAME='DGETRF'; D='results/<campaign>'; O='plots'" scripts/plot-strides.gp
#
# D/summary-R-strides.dat (scripts/summarize-strides.py), columns:
#   1 n, 2 t_openblas, 3 gflops_openblas,
#   then for the row strides 1, 2, 42 of ulmBLAS: t gflops speedup smin smax
#   (4-8: R = 1, 9-13: R = 2, 14-18: R = 42)
# Writes O/R-strides-gflops.svg, O/R-strides-time.svg, O/R-strides-speedup.svg.
#
# Colors: categorical slots 1-4 in fixed order; every series also has its
# own point type (two of the colors are below 3:1 contrast on white).

U1  = "#2a78d6"     # ulmBLAS, row stride 1   (slot 1)
OB  = "#eb6834"     # OpenBLAS, column major  (slot 2)
U2  = "#1baf7a"     # ulmBLAS, row stride 2   (slot 3)
U42 = "#eda100"     # ulmBLAS, row stride 42  (slot 4)
INK = "#52514e"

if (!exists("SUB")) SUB = "Xeon Platinum 8358 Ice Lake, one socket, 32 threads"

set terminal svg size 900,520 dynamic enhanced font "Helvetica,13" background rgb "white"
set border 3 lw 1.0 lc rgb INK
set tics nomirror tc rgb INK
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key top left Left reverse box lc rgb "#dcdcdc" opaque spacing 1.25 width 1
set xlabel "m = n"
set xrange [0:4050]
set xtics 0,500,4000
F = D."/summary-".R."-strides.dat"

# GFLOP/s
set output O."/".R."-strides-gflops.svg"
set title NAME.", ".SUB.": effective GFLOP/s"
set ylabel "GFLOP/s  (".(R eq "getrf" ? "2/3 n^3" : "1/3 n^3")." / time)"
set yrange [0:*]
plot F u 1:5  w lp lc rgb U1  lw 2 pt 7 ps 0.55 t "ulmBLAS, incRow = 1 (column major)", \
     F u 1:10 w lp lc rgb U2  lw 2 pt 9 ps 0.6  t "ulmBLAS, incRow = 2", \
     F u 1:15 w lp lc rgb U42 lw 2 pt 13 ps 0.6 t "ulmBLAS, incRow = 42", \
     F u 1:3  w lp lc rgb OB  lw 2 pt 5 ps 0.55 t "OpenBLAS 0.3.34 (column major)"

# time
set output O."/".R."-strides-time.svg"
set title NAME.", ".SUB.": time of one call"
set ylabel "seconds (log scale)"
set logscale y
set format y "10^{%L}"
set yrange [*:*]
plot F u 1:4  w lp lc rgb U1  lw 2 pt 7 ps 0.55 t "ulmBLAS, incRow = 1 (column major)", \
     F u 1:9  w lp lc rgb U2  lw 2 pt 9 ps 0.6  t "ulmBLAS, incRow = 2", \
     F u 1:14 w lp lc rgb U42 lw 2 pt 13 ps 0.6 t "ulmBLAS, incRow = 42", \
     F u 1:2  w lp lc rgb OB  lw 2 pt 5 ps 0.55 t "OpenBLAS 0.3.34 (column major)"
unset logscale y
set format y "%g"

# speedup over OpenBLAS (column major)
set output O."/".R."-strides-speedup.svg"
set title NAME.", ".SUB.": speedup over OpenBLAS (column major)"
set ylabel "time OpenBLAS / time ulmBLAS"
set yrange [0:*]
set key top right
plot 1 w l lc rgb INK dt 2 lw 1.2 t "equal time", \
     F u 1:6  w lp lc rgb U1  lw 2 pt 7 ps 0.55 t "incRow = 1 (median of the rounds)", \
     F u 1:11 w lp lc rgb U2  lw 2 pt 9 ps 0.6  t "incRow = 2", \
     F u 1:16 w lp lc rgb U42 lw 2 pt 13 ps 0.6 t "incRow = 42"
