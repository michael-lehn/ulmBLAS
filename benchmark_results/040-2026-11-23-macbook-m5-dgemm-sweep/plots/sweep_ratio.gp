load "_style.gp"
set terminal svg size 1150,560 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "sweep_ratio.svg"
set title "DGEMM, Apple M5, one performance core: speedup of ulmBLAS 0.4.2 over Apple Accelerate and BLIS PR #921\nratio of the medians of 5 interleaved rounds; above 1: ulmBLAS is faster" font ",14"
set xlabel "matrix size n = m = k"
set ylabel "speedup of ulmBLAS"
set xrange [0:4000]
set xtics 0, 500, 4000
set mxtics 5
set yrange [0.8:1.6]
set key inside right top vertical font ",11.5" spacing 1.05 box opaque
plot 1 with lines ls 9 notitle, \
     SW."squares.csv" using "n":"ulmblas_over_accelerate" with linespoints ls 6 title "over ".L6, \
     ""               using "n":"ulmblas_over_blis_sme"   with linespoints ls 7 title "over ".L7." (n = 50 … 350 off the scale: 1.7 to 9.1)"
