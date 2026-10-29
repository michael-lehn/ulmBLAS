load "_style.gp"
set terminal svg size 1150,640 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "sweep_small.svg"
set title "DGEMM, square matrices n = 50, 100, …, 500 (part of the sweep), Apple M5, one performance core\nmedians of 5 interleaved rounds, bands: minimum to maximum over the rounds; dotted: FMOPA peak (microbenchmark)" font ",14"
set xlabel "matrix size n = m = k"
set ylabel "performance [GFLOP/s]"
set xrange [0:500]
set xtics 0, 50, 500
set mxtics 2
set yrange [0:580]
set style fill transparent solid 0.18 noborder
set key inside right bottom vertical font ",11.5" spacing 1.05 box opaque
plot SMEPEAK with lines ls 9 notitle, \
     SW."squares.csv" using "n":"accelerate_min":"accelerate_max" with filledcurves lc rgb "#1baf7a" notitle, \
     ""               using "n":"blis_sme_min":"blis_sme_max"     with filledcurves lc rgb "#3a3a38" notitle, \
     ""               using "n":"published_min":"published_max"   with filledcurves lc rgb "#8c8c8c" notitle, \
     ""               using "n":"final_min":"final_max"           with filledcurves lc rgb "#2a78d6" notitle, \
     ""               using "n":"published"  with linespoints ls 1 title L1, \
     ""               using "n":"final"      with linespoints ls 5 title "ulmBLAS SME2, final (b38f3d5)", \
     ""               using "n":"accelerate" with linespoints ls 6 title L6, \
     ""               using "n":"blis_sme"   with linespoints ls 7 title L7
