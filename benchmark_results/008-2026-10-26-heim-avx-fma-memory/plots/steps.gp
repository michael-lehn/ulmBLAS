load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "steps.svg"
set title "DGEMM, square matrices n = 2 … 256 (unchanged up to 512): speedup of each ulmBLAS AVX/FMA change, Intel Core i5-12500, one core\n(time of the previous version / time of the version; medians of 5 interleaved rounds)" font ",14"
set xlabel "matrix size n"
set ylabel "speedup"
set xrange [0:260]
set xtics 0, 32, 256
set mxtics 2
set yrange [0.85:1.9]
set ytics 0.1
set key inside right top vertical font ",11.5" spacing 1.05 box opaque
plot 1 with lines ls 9 notitle, \
     D."dgemm_sweep.csv" using "n":"avx_pre_over_avx_pub"     with linespoints ls 2 title "panel grouping, constexpr parameters: 57e1c8a / e195bea", \
     ""                  using "n":"avx_cache_over_avx_pre"   with linespoints ls 3 title "per-thread buffer cache: 04cf17b / 57e1c8a", \
     ""                  using "n":"avx_cur_over_avx_cache"   with linespoints ls 7 title "control, same GEMM code: febb852 / 04cf17b", \
     ""                  using "n":"avx_cur_over_avx_pub"     with linespoints ls 4 title "overall: febb852 / e195bea"
