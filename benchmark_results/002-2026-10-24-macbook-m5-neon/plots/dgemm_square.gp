load "_style.gp"
set output "dgemm_square.svg"
set title "DGEMM, square matrices (m = n = k, column major, alpha = 1, beta = 1)\nApple M5 (MacBook Air), one performance core, double precision"
set xlabel "matrix size m = n = k"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set yrange [0:80]
set terminal svg size 960,680 dynamic enhanced font "Helvetica,15" background rgb "white"
set bmargin 10
set key at screen 0.5, screen 0.03 center bottom horizontal maxrows 2 width 2 nobox
set xtics 0, 512, 4096
set mxtics 2
#  FMA peak of the core: median of the measurements of run B (peak.c)
stats S."peak.csv" using (strcol("run") eq "B" ? column("gflops") : NaN) name "P" nooutput
plot P_median with lines ls 9 title sprintf("NEON FMA peak (measured, %.1f)", P_median), \
     S."dgemm_square.csv" using "n":"openblas"  with linespoints ls 6 title "OpenBLAS 0.3.34", \
     ""                   using "n":"blis"      with linespoints ls 5 title "BLIS 2.0", \
     ""                   using "n":"neon"      with linespoints ls 4 title "ulmBLAS NEON", \
     ""                   using "n":"neon_gcc"  with linespoints ls 3 title "ulmBLAS NEON (GCC 16)", \
     ""                   using "n":"base"      with linespoints ls 1 title "ulmBLAS before (reference kernel)"
