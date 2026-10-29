load "_style.gp"
set terminal svg size 1150,560 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "square.svg"
set multiplot layout 1,2 title "Square DGEMM on an Intel Core i5-12500 (heim), one core: ulmBLAS with OpenBLAS's HASWELL kernel\nmedians of 3 interleaved rounds (experiment iter1: baseline and iteration 1 measured together)" font ",14"
F = D."iter1.csv"
set xlabel "matrix size n (m = n = k)"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set yrange [0:65]
set key bottom right font ",11"
plot F using "n":(sq("ulmblas-openblas-haswell_4x8-iter1","gflops")) with linespoints ls 1 title "ulmBLAS + OpenBLAS kernel, iteration 1", \
     F using "n":(sq("ulmblas-openblas-haswell_4x8-base","gflops")) with linespoints ls 2 title "ulmBLAS + OpenBLAS kernel, baseline", \
     F using "n":(sq("openblas","gflops")) with linespoints ls 3 title "OpenBLAS 0.3.34 (HASWELL)", \
     F using "n":(sq("ulmblas-avx","gflops")) with linespoints ls 4 title "ulmBLAS AVX/FMA kernel", \
     F using "n":(sq("blis","gflops")) with linespoints ls 5 title "BLIS 2.0 (haswell)"
set ylabel "ratio to native OpenBLAS"
set yrange [0.8:1.1]
set arrow from graph 0, first 1 to graph 1, first 1 nohead ls 9
unset key
plot F using "n":(sq("ulmblas-openblas-haswell_4x8-iter1","vs_openblas")) with linespoints ls 1, \
     F using "n":(sq("ulmblas-openblas-haswell_4x8-base","vs_openblas")) with linespoints ls 2, \
     F using "n":(sq("ulmblas-avx","vs_openblas")) with linespoints ls 4, \
     F using "n":(sq("blis","vs_openblas")) with linespoints ls 5
unset multiplot
