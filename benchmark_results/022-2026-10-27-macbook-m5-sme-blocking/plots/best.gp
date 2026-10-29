load "_style.gp"
set terminal svg size 1150,980 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "best.svg"
set multiplot layout 2,2 title "DGEMM, square n: best MC/NC configuration per size against production (KC = 1024), Apple M5, one core\nbest = highest median of 19 (coarse, 3 rounds) or 7 (fine, 5 rounds) configurations, selected after the fact; bands: minimum to maximum over the rounds" font ",14"
F = D."best.csv"
c(e) = strcol("experiment") eq e
set ylabel "performance [GFLOP/s]"
set yrange [380:490]
set key bottom right font ",11.5" spacing 1.0
do for [r in "1 2"] {
  if (r eq "1") { set xrange [880:1220]; set xtics 900, 50, 1200; set arrow 1 from 1024, graph 0 to 1024, graph 1 nohead ls 9 } \
  else { set xrange [1880:2220]; set xtics 1900, 50, 2200; set arrow 1 from 2048, graph 0 to 2048, graph 1 nohead ls 9; unset key }
  set xlabel "matrix size n"
  plot F using "n":(c("coarse") ? column("best_min") : NaN):"best_max" with filledcurves lc rgb "#e8a33d" fs transparent solid 0.25 noborder notitle, \
       F using "n":(c("coarse") ? column("production_min") : NaN):"production_max" with filledcurves lc rgb "#2a78d6" fs transparent solid 0.20 noborder notitle, \
       F using "n":(c("coarse") ? column("best") : NaN) with linespoints ls 2 title "best (coarse)", \
       F using "n":(c("fine") ? column("best") : NaN) with points lc rgb "#c4553a" pt 5 ps 0.7 title "best (fine)", \
       F using "n":(c("coarse") ? column("production") : NaN) with linespoints ls 1 title "production (coarse)", \
       F using "n":(c("fine") ? column("production") : NaN) with points lc rgb "#2a78d6" pt 6 ps 0.8 title "production (fine)"
}
set ylabel "best / production"
set yrange [0.99:1.03]
set ytics 0.99, 0.01, 1.03
set key top right font ",11.5"
do for [r in "1 2"] {
  if (r eq "1") { set xrange [880:1220]; set xtics 900, 50, 1200; set arrow 1 from 1024, graph 0 to 1024, graph 1 nohead ls 9 } \
  else { set xrange [1880:2220]; set xtics 1900, 50, 2200; set arrow 1 from 2048, graph 0 to 2048, graph 1 nohead ls 9; unset key }
  set arrow 2 from graph 0, first 1 to graph 1, first 1 nohead ls 9
  plot F using "n":(c("coarse") ? column("ratio") : NaN) with linespoints ls 2 title "coarse", \
       F using "n":(c("fine") ? column("ratio") : NaN) with linespoints lc rgb "#c4553a" lw 1.6 pt 5 ps 0.6 title "fine"
}
unset multiplot
