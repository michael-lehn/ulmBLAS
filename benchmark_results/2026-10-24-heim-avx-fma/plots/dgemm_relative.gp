load "_style.gp"
set output "dgemm_relative.svg"
set title "DGEMM relative to OpenBLAS 0.3.34\nsquare matrices, alpha = 1, beta = 1, Intel Core i5-12500 (heim), one core"
set xlabel "matrix size m = n = k"
set ylabel "performance / performance of OpenBLAS"
set logscale x 2
set xrange [5:5000]
set yrange [0:1.4]
set ytics 0.2
set key inside right center
set xtics (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
plot 1 with lines ls 9 notitle, \
     S."dgemm_square.csv" using "n":"blis_over_openblas"      with linespoints ls 5 title "BLIS 2.0", \
     ""                   using "n":"avx_mc288_over_openblas" with linespoints ls 4 title "ulmBLAS AVX/FMA (MC=288)", \
     ""                   using "n":"sse_over_openblas"       with linespoints ls 2 title "ulmBLAS SSE"
