load "_style.gp"
set terminal svg size 1150,560 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "sweep.svg"
set multiplot layout 1,2 title "Square DGEMM n = 8 ... 250 on an Intel Xeon Platinum 8358 (Ice Lake), one core: ulmBLAS (df58cfa, OpenBLAS's SKYLAKEX kernel, 16 x 12 tiles) and OpenBLAS 0.3.34\nmedians of 3 rounds; dotted line: n = 100, the largest n for OpenBLAS's small-matrix kernel (m n k <= 10^6)" font ",13"
F = D."sweep.csv"
set xlabel "matrix size n (m = n = k)"
set xrange [0:256]
set xtics 0, 32, 256
set arrow 1 from 100, graph 0 to 100, graph 1 nohead ls 9
set ylabel "performance [GFLOP/s]"
set yrange [0:95]
set key bottom right font ",11"
plot F using "n":"ulmblas" with linespoints ls 1 title "ulmBLAS", \
     F using "n":"openblas" with linespoints ls 3 title "OpenBLAS"
set ylabel "ulmBLAS / OpenBLAS"
set yrange [0:1.4]
set arrow 2 from graph 0, first 1 to graph 1, first 1 nohead ls 9
full(c) = column("full_tiles")==1 ? column(c) : NaN
set key bottom right font ",11"
plot F using "n":"ratio" with linespoints ls 1 title "ratio", \
     F using "n":(full("ratio")) with points pt 5 ps 1.3 lc rgb "#c4553a" title "n multiple of 16 and 12 (full tiles only)"
unset multiplot
