load "_style.gp"
set output "dgemm_speedup.svg"
set title "DGEMM: speedup of the AVX/FMA micro kernel over the SSE micro kernel\nsquare matrices, alpha = 1, beta = 1, Intel Core i5-12500 (heim), one core"
set xlabel "matrix size m = n = k"
set ylabel "speedup (GFLOP/s AVX/FMA / GFLOP/s SSE)"
set logscale x 2
set xrange [5:5000]
set yrange [0:3.5]
set ytics 0.5
set key inside right bottom
set xtics (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
plot 1 with lines ls 9 notitle, \
     S."dgemm_square.csv" using "n":"avx_mc288_over_sse" with linespoints ls 4 title "AVX/FMA (MC=288) / SSE", \
     ""                   using "n":"avx_mc72_over_sse"  with linespoints ls 3 title "AVX/FMA (MC=72, run A) / SSE"
