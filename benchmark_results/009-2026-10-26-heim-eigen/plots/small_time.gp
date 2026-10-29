load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "small_time.svg"
set title "DGEMM C = AB + C, time per call for n = 1 … 32, Intel Core i5-12500, one core\nmedians of 5 interleaved rounds (lower is better)" font ",14"
set xlabel "matrix size n"
set ylabel "time per call [µs]"
set xrange [0:33]
set xtics 0, 4, 32
set mxtics 4
set logscale y
set yrange [0.005:3]
set key inside right bottom vertical font ",11.5" spacing 1.05 box opaque
plot D."dgemm_fine.csv" using "n":(column("n") <= 32 ? column("avx_cur_us")  : NaN) with linespoints ls 4 title L4, \
     ""                 using "n":(column("n") <= 32 ? column("blis_us")     : NaN) with linespoints ls 7 title L7, \
     ""                 using "n":(column("n") <= 32 ? column("openblas_us") : NaN) with linespoints ls 8 title L8, \
     ""                 using "n":(column("n") <= 32 ? column("eigen_us")    : NaN) with linespoints ls 10 title L10
