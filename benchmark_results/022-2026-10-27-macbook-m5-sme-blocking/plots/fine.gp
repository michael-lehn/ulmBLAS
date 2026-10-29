load "_style.gp"
set terminal svg size 1150,600 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "fine.svg"
set multiplot layout 1,2 title "DGEMM, square n: fine grid of MC and NC (KC = 1024), Apple M5, one core\nmedians of 5 interleaved rounds; production MC, KC, NC = 384, 1024, 1024 (thick)" font ",14"
set ylabel "performance [GFLOP/s]"
set xlabel "matrix size n"
set yrange [395:485]
F = D."fine.csv"
s(v) = strcol("config") eq v ? column("gflops") : NaN
set xrange [990:1210]
set xtics 1000, 50, 1200
set arrow 1 from 1024, graph 0 to 1024, graph 1 nohead ls 9
set key bottom right font ",11" spacing 1.0
plot F using "n":(s("b352_1024_1024")) with linespoints ls 2 title "MC 352, NC 1024", \
     F using "n":(s("b384_1024_1216")) with linespoints ls 3 title "MC 384, NC 1216", \
     F using "n":(s("b352_1024_1216")) with linespoints ls 4 title "MC 352, NC 1216", \
     F using "n":(s("b384_1024_2208")) with linespoints ls 5 title "MC 384, NC 2208", \
     F using "n":(s("b352_1024_2208")) with linespoints ls 6 title "MC 352, NC 2208", \
     F using "n":(s("b384_1024_4096")) with linespoints lc rgb "#b0b0b0" lw 1.6 dt 2 pt 6 ps 0.6 title "MC 384, NC 4096", \
     F using "n":(s(P)) with linespoints ls 1 title "production"
set xrange [1990:2210]
set xtics 2000, 50, 2200
set arrow 1 from 2048, graph 0 to 2048, graph 1 nohead ls 9
unset key
replot
unset multiplot
