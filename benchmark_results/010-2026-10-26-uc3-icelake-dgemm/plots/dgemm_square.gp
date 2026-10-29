load "_style.gp"
load "_labels.gp"
set terminal svg size 1150,660 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "dgemm_square.svg"
set title "DGEMM C = AB + C, square matrices n = 7 … 4096, ".CPU.", one core\nmedians of 3 interleaved rounds; dashed: FMA peak of the core" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set xtics 0, 512, 4096
set mxtics 2
set yrange [0:YMAX]
set key outside right center vertical font ",11" spacing 1.05 box opaque
plot D."dgemm_square.csv" using "n":"ulmblas"      with linespoints ls 1 title L1, \
     ""                   using "n":"blis_auto"    with linespoints ls 2 title L2, \
     ""                   using "n":"blis_haswell" with linespoints ls 3 title L3, \
     ""                   using "n":"openblas"     with linespoints ls 4 title L4, \
     ""                   using "n":"mkl"          with linespoints ls 5 title L5, \
     ""                   using "n":"aocl"         with linespoints ls 6 title L6, \
     ""                   using "n":"eigen"        with linespoints ls 7 title L7, \
     PYMM with lines ls 9 title "FMA peak, 256 bit", \
     (PZMM > PYMM*1.05 ? PZMM : 1/0) with lines ls 9 title "FMA peak, 512 bit"
