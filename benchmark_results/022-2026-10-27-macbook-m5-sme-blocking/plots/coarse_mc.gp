load "_style.gp"
set terminal svg size 1150,600 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "coarse_mc.svg"
set multiplot layout 1,2 title "DGEMM, square n: performance against MC (KC = 1024, NC = 1024), Apple M5, one core\nmedians of 3 interleaved rounds; production MC = 384 (dotted)" font ",14"
set xlabel "MC"
set ylabel "performance [GFLOP/s]"
set xrange [160:800]
set xtics 192, 64, 768
set yrange [360:490]
set key bottom right font ",11.5" spacing 1.0
set arrow 1 from 384, graph 0 to 384, graph 1 nohead ls 9
F = D."coarse.csv"
s(n) = column("NC")==1024 && column("KC")==1024 && column("n")==n ? column("gflops") : NaN
N = "1000 1024 1025 1050 1100"
plot for [i=1:5] F using "MC":(s(int(word(N,i)))) with linespoints lw 1.8 lc rgb SC[i] pt SP[i] ps 0.6 title "n = ".word(N,i)
N = "2000 2048 2049 2050 2100"
replot
unset multiplot
