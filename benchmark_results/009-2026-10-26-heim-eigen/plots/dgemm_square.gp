load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "dgemm_square.svg"
set title "DGEMM C = AB + C, square matrices n = 7 … 4096, Intel Core i5-12500, one core\nmedians of 3 interleaved rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set xtics 0, 512, 4096
set mxtics 2
set yrange [0:66]
set key inside right center vertical font ",11.5" spacing 1.05 box opaque
plot D."dgemm_square.csv" using "n":"avx_cur"  with linespoints ls 4 title L4, \
     ""                   using "n":"blis"     with linespoints ls 7 title L7, \
     ""                   using "n":"openblas" with linespoints ls 8 title L8, \
     ""                   using "n":"eigen"    with linespoints ls 10 title L10
