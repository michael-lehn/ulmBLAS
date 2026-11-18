load "_style.gp"
set terminal svg size 1250,680 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "s_square.svg"
set title "sgemm, square matrices, column major, float: ulmBLAS::gemm\n".CPU."\nmedian of 5 rounds (each the median of 5 samples of >= 20 ms)" font ",14"
set xlabel "matrix size n (m = n = k)"
set ylabel "performance [GFLOP/s]"
set logscale x 2
set yrange [0:*]
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
plot D."s_square.csv" using "n":"master" with linespoints pt 7 ps 0.6 lc rgb "#52514e" lw 2 title "master (before the redesign)", \
     D."s_square.csv" using "n":"generic" with linespoints pt 7 ps 0.6 lc rgb "#2a78d6" lw 2 title "redesign, hand-written table", \
     D."s_square.csv" using "n":"tuned" with linespoints pt 7 ps 0.6 lc rgb "#e34948" lw 2 title "redesign, table of make tune"
