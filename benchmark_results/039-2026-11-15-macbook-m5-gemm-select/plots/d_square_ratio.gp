load "_style.gp"
set terminal svg size 1250,680 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "d_square_ratio.svg"
set title "dgemm, square matrices: relative to master\n".CPU font ",14"
set xlabel "matrix size n (m = n = k)"
set ylabel "performance relative to master"
set logscale x 2
set yrange [0.8:1.3]
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
plot 1 with lines lc rgb "#52514e" dt 2 title "master", \
     D."d_square.csv" using "n":"generic_ratio" with linespoints pt 7 ps 0.6 lc rgb "#2a78d6" lw 2 title "redesign, hand-written table", \
     D."d_square.csv" using "n":"tuned_ratio" with linespoints pt 7 ps 0.6 lc rgb "#e34948" lw 2 title "redesign, table of make tune"
