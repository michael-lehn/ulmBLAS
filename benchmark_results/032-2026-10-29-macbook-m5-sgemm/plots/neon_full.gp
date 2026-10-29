load "_style.gp"
set terminal svg size 1250,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "neon_full.svg"
set title "SGEMM without SME2: NEON kernels, m = n = k = 50 ... 4000, single-threaded\n".CPU."\nmedians of 3 interleaved rounds (2mnk flops)" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:4050]
set xtics 500
set yrange [0:140]
set ytics 20
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
plot for [i in "2 4 5"] D."gemm.csv" using "n":V[int(i)] with linespoints \
         lc rgb C[int(i)] lw (K[int(i)] ? 2.2 : 1.6) dt (K[int(i)] ? 1 : 2) pt P[int(i)] pi 8 ps 0.75 title T[int(i)]
