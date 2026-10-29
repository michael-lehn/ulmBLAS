load "_style.gp"
load "_labels.gp"
set terminal svg size 1150,660 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "sweep.svg"
set title "DGEMM C = AB + C, square matrices n = 2 … 512, ".CPU.", one core\nmedians of 5 interleaved rounds, bands: minimum to maximum over the rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:520]
set xtics 0, 64, 512
set mxtics 2
set yrange [0:YMAX]
set key outside right center vertical font ",11" spacing 1.05 box opaque
plot D."dgemm_sweep.csv" using "n":"ulmblas_min":"ulmblas_max" with filledcurves lc rgb "#2a78d6" fs transparent solid 0.3 noborder notitle, \
     ""                  using "n":"ulmblas"      with linespoints ls 1 title L1, \
     ""                  using "n":"blis_auto"    with linespoints ls 2 title L2, \
     ""                  using "n":"blis_haswell" with linespoints ls 3 title L3, \
     ""                  using "n":"openblas"     with linespoints ls 4 title L4, \
     ""                  using "n":"mkl"          with linespoints ls 5 title L5, \
     ""                  using "n":"aocl"         with linespoints ls 6 title L6, \
     ""                  using "n":"eigen"        with linespoints ls 7 title L7
