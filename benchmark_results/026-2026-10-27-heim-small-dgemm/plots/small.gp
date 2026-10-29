load "_style.gp"
set terminal svg size 1150,560 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "small.svg"
set multiplot layout 1,2 title "Square DGEMM n = 2 ... 512 on an Intel Core i5-12500 (heim), one core: ulmBLAS with OpenBLAS's HASWELL kernel (ae96628, 4 x 8 tiles)\nand native OpenBLAS 0.3.34 (HASWELL, no small-matrix kernel), medians of 3 interleaved rounds" font ",13"
F = D."small.csv"
set xlabel "matrix size n (m = n = k)"
set xrange [0:520]
set ylabel "performance [GFLOP/s]"
set yrange [0:62]
set key bottom right font ",11"
plot F using "n":(sq("ulmblas-openblas-haswell_4x8-iter1","gflops")) with linespoints ls 1 title "ulmBLAS + OpenBLAS kernel", \
     F using "n":(sq("openblas","gflops")) with linespoints ls 3 title "OpenBLAS 0.3.34 (HASWELL)", \
     F using "n":(sq("ulmblas-avx","gflops")) with linespoints ls 4 title "ulmBLAS AVX/FMA kernel", \
     F using "n":(sq("blis","gflops")) with linespoints ls 5 title "BLIS 2.0 (haswell)"
set xrange [0:136]
set ylabel "ratio to native OpenBLAS"
set yrange [0:1.2]
set arrow from graph 0, first 1 to graph 1, first 1 nohead ls 9
full(c) = (strcol("variant") eq "ulmblas-openblas-haswell_4x8-iter1" && int(column("n"))%8==0) ? column(c) : NaN
plot F using "n":(sq("ulmblas-openblas-haswell_4x8-iter1","vs_openblas")) with linespoints ls 1 title "ulmBLAS + OpenBLAS kernel", \
     F using "n":(full("vs_openblas")) with points pt 5 ps 1.2 lc rgb "#c4553a" title "n multiple of 8 (full 4 x 8 tiles)"
unset multiplot
