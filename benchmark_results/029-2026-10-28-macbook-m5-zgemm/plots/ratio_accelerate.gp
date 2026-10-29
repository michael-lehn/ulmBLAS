load "_style.gp"
set terminal svg size 1250,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "ratio_accelerate.svg"
set title "ulmBLAS relative to Apple Accelerate, ZGEMM m = n = k, single-threaded\n".CPU."\nmedians of the per-round ratios; bands: minimum to maximum over the rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance relative to Apple Accelerate"
set xrange [0:4050]
set xtics 500
set yrange [0:2.3]
set ytics 0.2
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
set arrow from 0,1 to 4050,1 nohead lc rgb "#0b0b0b" lw 1.2
plot for [i=1:2] D."ratios.csv" using "n":(column(V[i]."_over_accelerate_min")):(column(V[i]."_over_accelerate_max")) \
         with filledcurves fs transparent solid 0.18 noborder lc rgb C[i] notitle, \
     for [i=1:2] D."ratios.csv" using "n":V[i]."_over_accelerate" with linespoints \
         lc rgb C[i] lw 2.2 pt P[i] pi 8 ps 0.75 title T[i]
