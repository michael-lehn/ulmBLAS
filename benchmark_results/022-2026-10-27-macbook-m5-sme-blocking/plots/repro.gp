load "_style.gp"
set terminal svg size 1150,560 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "repro.svg"
set multiplot layout 1,2 title "DGEMM, square n, production configuration (MC, KC, NC = 384, 1024, 1024), Apple M5, one core\nmedians of 3 interleaved rounds, band: minimum to maximum" font ",14"
set ylabel "performance [GFLOP/s]"
set xlabel "matrix size n"
set yrange [380:490]
set xrange [890:1210]
set xtics 900, 50, 1200
set arrow 1 from 1024, graph 0 to 1024, graph 1 nohead ls 9
plot D."repro.csv" using "n":"gflops_min":"gflops_max" with filledcurves lc rgb "#2a78d6" fs transparent solid 0.25 noborder notitle, \
     ""            using "n":"gflops" with linespoints ls 1 notitle
set xrange [1890:2210]
set xtics 1900, 50, 2200
set arrow 1 from 2048, graph 0 to 2048, graph 1 nohead ls 9
plot D."repro.csv" using "n":"gflops_min":"gflops_max" with filledcurves lc rgb "#2a78d6" fs transparent solid 0.25 noborder notitle, \
     ""            using "n":"gflops" with linespoints ls 1 notitle
unset multiplot
