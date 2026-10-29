load "_style.gp"
set terminal svg size 1150,800 dynamic enhanced font "Helvetica,13" background rgb "white"
set output "level3.svg"
set multiplot layout 2,2 title "Level 3 (n x n, left, lower, NoTrans), Intel Core i5-12500, one core: medians of 3 interleaved rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [400:3200]
set xtics 512, 512, 3072
set yrange [40:62]
set key inside right bottom vertical font ",10" spacing 1.0 box opaque
R = "dsyrk dsymm dtrmm dtrsm"
do for [i=1:4] {
    set title word(R, i) font ",13"
    F = D."level3_".word(R, i).".csv"
    plot F using "n":"avx_pub"   with linespoints ls 1 title L1, \
         F using "n":"avx_pre"   with linespoints ls 2 title L2, \
         F using "n":"avx_cache" with linespoints ls 3 title L3, \
         F using "n":"avx_cur"   with linespoints ls 4 title L4, \
         F using "n":"blis"      with linespoints ls 7 title L7, \
         F using "n":"openblas"  with linespoints ls 8 title L8
    unset key
}
unset multiplot
