load "_style.gp"
set output "dgemm_square.svg"
set title "DGEMM, square matrices (m = n = k, column major, alpha = 1, beta = 1)\nIntel Core i5-12500 (heim), one core, double precision"
set xlabel "matrix size m = n = k"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set yrange [0:70]
set terminal svg size 960,680 dynamic enhanced font "Helvetica,15" background rgb "white"
set bmargin 10
set key at screen 0.5, screen 0.03 center bottom horizontal maxrows 2 width 2 nobox
set xtics 0, 512, 4096
set mxtics 2
plot S."dgemm_square.csv" using "n":"openblas"  with linespoints ls 6 title "OpenBLAS 0.3.34", \
     ""                   using "n":"blis"      with linespoints ls 5 title "BLIS 2.0", \
     ""                   using "n":"avx_mc288" with linespoints ls 4 title "ulmBLAS AVX/FMA (MC=288)", \
     ""                   using "n":"avx_mc72"  with linespoints ls 3 title "ulmBLAS AVX/FMA (MC=72, run A)", \
     ""                   using "n":"sse_mc384" with linespoints ls 2 title "ulmBLAS SSE", \
     ""                   using "n":"ref_mc384" with linespoints ls 1 title "ulmBLAS reference (run A)"
