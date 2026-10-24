load "_style.gp"
set output "dgemm_speedup.svg"
set title "DGEMM: speedup of the NEON micro kernel over the reference kernel\nsquare matrices, alpha = 1, beta = 1, Apple M5 (MacBook Air), one performance core"
set xlabel "matrix size m = n = k"
set ylabel "speedup"
set logscale x 2
set xrange [5:5000]
set yrange [0:2.5]
set ytics 0.5
set key inside right bottom
set xtics (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
plot 1 with lines ls 9 notitle, \
     S."dgemm_square.csv" using "n":"neon_over_base" with linespoints ls 4 title "NEON / ulmBLAS before (reference kernel)", \
     ""                   using "n":"neon_over_blis" with linespoints ls 5 title "NEON / BLIS 2.0"
