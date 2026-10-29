load "_style.gp"
set terminal svg size 1150,600 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "coarse_nc.svg"
set multiplot layout 1,2 title "DGEMM, square n: performance against NC (MC = 384, KC = 1024), Apple M5, one core\nmedians of 3 interleaved rounds; production NC = 1024 (dotted)" font ",14"
set xlabel "NC"
set ylabel "performance [GFLOP/s]"
set xrange [400:4200]
set xtics 512, 512, 4096
set yrange [360:490]
set key bottom right font ",11.5" spacing 1.0
set arrow 1 from 1024, graph 0 to 1024, graph 1 nohead ls 9
F = D."coarse.csv"
s(n) = column("MC")==384 && column("KC")==1024 && column("n")==n ? column("gflops") : NaN
N = "1000 1024 1025 1050 1100"
plot for [i=1:5] F using "NC":(s(int(word(N,i)))) with linespoints lw 1.8 lc rgb SC[i] pt SP[i] ps 0.6 title "n = ".word(N,i)
N = "2000 2048 2049 2050 2100"
replot
unset multiplot
