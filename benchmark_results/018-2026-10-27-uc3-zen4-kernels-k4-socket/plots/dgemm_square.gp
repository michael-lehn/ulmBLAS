load "_style.gp"
load "_labels.gp"
set terminal svg size 1250,680 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "dgemm_square.svg"
set title "DGEMM C = AB + C, square matrices n = 7 … 4096, ".CPU.", one core\nmedians of interleaved rounds; solid: micro kernels in ulmBLAS, dashed: native libraries; grey: FMA peak" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set xtics 0, 512, 4096
set yrange [0:YMAX]
set key outside right center vertical font ",10.5" spacing 1.0 box opaque
plot for [i=1:|V|] D."dgemm_square.csv" using "n":V[i] with linespoints \
         lc rgb color(i) lw (K[i] ? 2.4 : 1.8) dt (K[i] ? 1 : 2) pt (K[i] ? 7 : 4) ps 0.55 title T[i], \
     PYMM with lines lc rgb "#909090" lw 1.2 dt (6,4) notitle, \
     (PZMM > PYMM*1.05 ? PZMM : 1/0) with lines lc rgb "#909090" lw 1.2 dt (6,4) notitle
