load "_style.gp"
load "_labels.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "thin_k.svg"
set title "DGEMM C = AB + C, m x n x k with small k (beta = 1), ".CPU.", one core\nmedians of 3 interleaved rounds" font ",14"
set ylabel "performance [GFLOP/s]"
set yrange [0:YMAX]
set style data histograms
set style histogram clustered gap 1.5
set style fill solid 0.85 noborder
set boxwidth 0.9
set xtics nomirror rotate by 0
set key outside right center vertical font ",11" box opaque
plot D."dgemm_thin_k.csv" using "ulmblas":xtic(sprintf("%dx%dx%d", column(1), column(2), column(3))) lc rgb "#2a78d6" title L1, \
     "" using "blis_auto"    lc rgb "#3a3a38" title L2, \
     "" using "blis_haswell" lc rgb "#9a9a96" title L3, \
     "" using "openblas"     lc rgb "#1baf7a" title L4, \
     "" using "mkl"          lc rgb "#e8871e" title L5, \
     "" using "aocl"         lc rgb "#c8323c" title L6, \
     "" using "eigen"        lc rgb "#9b3fb5" title L7
