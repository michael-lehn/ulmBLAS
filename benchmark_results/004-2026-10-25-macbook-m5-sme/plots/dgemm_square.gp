load "_style.gp"
set terminal svg size 960,700 dynamic enhanced font "Helvetica,15" background rgb "white"
set output "dgemm_square.svg"
set title "DGEMM, square matrices (m = n = k, column major, alpha = 1, beta = 1)\nApple M5 (MacBook Air), one performance core, double precision"
set xlabel "matrix size m = n = k"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set yrange [0:580]
set bmargin 10
set key at screen 0.5, screen 0.03 center bottom horizontal maxrows 3 width 2 nobox
set xtics 0, 512, 4096
set mxtics 2
plot PEAK_max with lines ls 9 title sprintf("SME2 FMOPA peak (measured, %.0f)", PEAK_max), \
     S."dgemm_square.csv" using "case":"sme"        with linespoints ls 1 title "ulmBLAS SME2 (32 x 16)", \
     ""                   using "case":"blis_sme"   with linespoints ls 2 title "BLIS PR #921 SME2", \
     ""                   using "case":"accelerate" with linespoints ls 3 title "Apple Accelerate", \
     ""                   using "case":"neon"       with linespoints ls 4 title "ulmBLAS NEON (6 x 8)", \
     ""                   using "case":"blis"       with linespoints ls 5 title "BLIS 2.0 NEON", \
     ""                   using "case":"openblas"   with linespoints ls 6 title "OpenBLAS 0.3.34"
