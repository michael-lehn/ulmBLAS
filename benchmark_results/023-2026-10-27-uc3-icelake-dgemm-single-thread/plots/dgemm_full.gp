load "_style.gp"
set terminal svg size 1250,680 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "dgemm_full.svg"
set title "DGEMM C = AB + C, m = n = k = 50 ... 4000, double precision, single-threaded\n".CPU."\nmedians of 3 interleaved rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:4050]
set xtics 500
set yrange [0:112]
set ytics 10
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
plot for [i=1:|V|] D."dgemm.csv" using "n":V[i] with linespoints \
         lc rgb C[i] lw (K[i] ? 2.2 : 1.6) dt (K[i] ? 1 : 2) pt P[i] pi 8 ps 0.75 title T[i], \
     PZMM with lines lc rgb "#52514e" dt 3 lw 1.2 title "FMA peak, 512 bit (105.2)", \
     PYMM with lines lc rgb "#52514e" dt 4 lw 1.2 title "FMA peak, 256 bit (54.2)"
