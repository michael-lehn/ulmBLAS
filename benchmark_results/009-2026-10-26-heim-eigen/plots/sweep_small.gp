load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "sweep_small.svg"
set title "DGEMM C = AB + C, square matrices, every n = 1 … 64, Intel Core i5-12500, one core\nmedians of 5 interleaved rounds, bands: minimum to maximum over the rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:65]
set xtics 0, 8, 64
set mxtics 4
set yrange [0:60]
set key inside right bottom vertical font ",11.5" spacing 1.05 box opaque
plot D."dgemm_fine.csv" using "n":"openblas_min":"openblas_max" with filledcurves lc rgb "#1baf7a" fs transparent solid 0.25 noborder notitle, \
     ""                 using "n":"blis_min":"blis_max"         with filledcurves lc rgb "#3a3a38" fs transparent solid 0.25 noborder notitle, \
     ""                 using "n":"avx_cur_min":"avx_cur_max"   with filledcurves lc rgb "#2a78d6" fs transparent solid 0.3 noborder notitle, \
     ""                 using "n":"eigen_min":"eigen_max"       with filledcurves lc rgb "#9b3fb5" fs transparent solid 0.3 noborder notitle, \
     ""                 using "n":"avx_cur"  with linespoints ls 4 title L4, \
     ""                 using "n":"blis"     with linespoints ls 7 title L7, \
     ""                 using "n":"openblas" with linespoints ls 8 title L8, \
     ""                 using "n":"eigen"    with linespoints ls 10 title L10
