load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "sweep_small.svg"
set title "DGEMM, square matrices n = 2 … 128: the ulmBLAS AVX/FMA versions, Intel Core i5-12500, one core\nmedians of 5 interleaved rounds, bands: minimum to maximum over the rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:130]
set xtics 0, 16, 128
set mxtics 2
set yrange [0:62]
set key inside right bottom vertical font ",11.5" spacing 1.05 box opaque
plot D."dgemm_sweep.csv" using "n":"openblas_min":"openblas_max" with filledcurves lc rgb "#1baf7a" fs transparent solid 0.25 noborder notitle, \
     ""                  using "n":"blis_min":"blis_max"         with filledcurves lc rgb "#3a3a38" fs transparent solid 0.25 noborder notitle, \
     ""                  using "n":"avx_pub_min":"avx_pub_max"   with filledcurves lc rgb "#8c8c8c" fs transparent solid 0.3 noborder notitle, \
     ""                  using "n":"avx_cur_min":"avx_cur_max"   with filledcurves lc rgb "#2a78d6" fs transparent solid 0.3 noborder notitle, \
     ""                  using "n":"avx_pub"   with linespoints ls 1 title L1, \
     ""                  using "n":"avx_pre"   with linespoints ls 2 title L2, \
     ""                  using "n":"avx_cache" with linespoints ls 3 title L3, \
     ""                  using "n":"avx_cur"   with linespoints ls 4 title L4, \
     ""                  using "n":"blis"      with linespoints ls 7 title L7, \
     ""                  using "n":"openblas"  with linespoints ls 8 title L8
