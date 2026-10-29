load "_style.gp"
set terminal svg size 1200,640 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "dgemm_shapes.svg"
set title "DGEMM, rectangular and thin matrices (column major, alpha = 1, beta = 1)\nApple M5 (MacBook Air), one performance core"
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.95 border rgb "white"
set boxwidth 0.95
set grid noxtics
set xtics rotate by -35 font ",12"
set yrange [0:640]
set xlabel "m x n x k"
set ylabel "performance [GFLOP/s]"
set key inside top center horizontal maxrows 1 font ",12" width 2 nobox
plot S."dgemm_shapes.csv" using "sme":xtic(1) lc rgb "#2a78d6" title "ulmBLAS SME2", \
     "" using "blis_sme"   lc rgb "#eb6834" title "BLIS PR #921 SME2", \
     "" using "accelerate" lc rgb "#1baf7a" title "Apple Accelerate", \
     "" using "neon"       lc rgb "#eda100" title "ulmBLAS NEON", \
     PEAK_max with lines ls 9 notitle
