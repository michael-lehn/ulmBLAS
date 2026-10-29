load "_style.gp"
load "_labels.gp"
set terminal svg size 1250,680 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "dgemm_large.svg"
set title "DGEMM C = AB + C, m = n = k ≥ 1000, ".CPU.", one core\nmedians of interleaved rounds, bars: minimum to maximum over the rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [900:4200]
set xtics (1000, 1500, 2000, 3000, 4000)
set yrange [0:YMAX]
set key outside right center vertical font ",10.5" spacing 1.0 box opaque
plot for [i=1:|V|] D."dgemm_square.csv" using "n":V[i]:(column(V[i]."_min")):(column(V[i]."_max")) \
         every ::0 with yerrorlines lc rgb color(i) lw (K[i] ? 2.4 : 1.8) dt (K[i] ? 1 : 2) pt (K[i] ? 7 : 4) ps 0.55 title T[i]
