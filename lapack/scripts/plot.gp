# Plots of the ulmlapack campaign (gnuplot >= 5.4).
#
#   gnuplot -e "R='getrf'; NAME='DGETRF'; D='results/<campaign>'; O='plots'" scripts/plot.gp
#
# D/summary-R.dat columns: n t_ulm t_ob gf_ulm gf_ob speedup sp_min sp_max ...
# Writes O/R-gflops.svg, O/R-time.svg, O/R-speedup.svg.

ULM = "#2a78d6"     # ulmBLAS  (categorical slot 1)
OB  = "#eb6834"     # OpenBLAS (categorical slot 2)
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
F = D."/summary-".R.".dat"

# GFLOP/s
set output O."/".R."-gflops.svg"
set title NAME.", ".SUB.": effective GFLOP/s"
set ylabel "GFLOP/s  (".(R eq "getrf" ? "2/3 n^3" : "1/3 n^3")." / time)"
set yrange [0:*]
plot F u 1:4 w lp lc rgb ULM lw 2 pt 7 ps 0.55 t "ulmBLAS (OpenMP tasks)", \
     F u 1:5 w lp lc rgb OB  lw 2 pt 5 ps 0.55 t "OpenBLAS 0.3.34"

# time
set output O."/".R."-time.svg"
set title NAME.", ".SUB.": time of one call"
set ylabel "seconds (log scale)"
set logscale y
set format y "10^{%L}"
set yrange [*:*]
plot F u 1:2 w lp lc rgb ULM lw 2 pt 7 ps 0.55 t "ulmBLAS (OpenMP tasks)", \
     F u 1:3 w lp lc rgb OB  lw 2 pt 5 ps 0.55 t "OpenBLAS 0.3.34"
unset logscale y
set format y "%g"

# speedup
set output O."/".R."-speedup.svg"
set title NAME.", ".SUB.": speedup of ulmBLAS over OpenBLAS"
set ylabel "time OpenBLAS / time ulmBLAS"
set yrange [0:*]
set key top right
plot F u 1:7:8 w filledcurves lc rgb "#cde2fb" t "range over the rounds", \
     1 w l lc rgb INK dt 2 lw 1.2 t "equal time", \
     F u 1:6 w lp lc rgb ULM lw 2 pt 7 ps 0.55 t "median over the rounds"
