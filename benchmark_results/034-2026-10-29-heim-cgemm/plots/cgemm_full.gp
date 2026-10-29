load "_style.gp"
set terminal svg size 1250,680 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "cgemm_full.svg"
set title "CGEMM C = AB + C, m = n = k = 50 ... 4000, complex float, single-threaded\n".CPU."\nmedians of 3 interleaved rounds (8mnk flops)" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:4050]
set xtics 500
set yrange [0:130]
set ytics 10
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
plot for [i=1:6] D."gemm.csv" using "n":V[i] with linespoints \
         lc rgb C[i] lw 2.2 dt 1 pt P[i] pi 8 ps 0.75 title T[i], \
     for [i=7:|V|] D."gemm.csv" using "n":V[i] with linespoints \
         lc rgb C[i] lw 1.6 dt 2 pt P[i] pi 8 ps 0.75 title T[i]
