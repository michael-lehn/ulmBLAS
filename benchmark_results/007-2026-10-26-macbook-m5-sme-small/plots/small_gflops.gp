load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "small_gflops.svg"
set title "DGEMM, small square matrices (m = n = k = 2 … 64), Apple M5, one performance core\nmedians of 3 interleaved rounds; dotted: NEON FMA peak (microbenchmark)" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:66]
set xtics 0, 8, 64
set mxtics 2
set yrange [0:400]
set key inside left top vertical font ",11.5" spacing 1.05 box opaque
plot NEONPEAK with lines ls 9 notitle, \
     S."squares.csv" using "n":"published" with linespoints ls 1 title L1, \
     ""              using "n":"packtail"  with linespoints ls 2 title L2, \
     ""              using "n":"cache"     with linespoints ls 3 title L3, \
     ""              using "n":"direct"    with linespoints ls 4 title L4, \
     ""              using "n":"final"     with linespoints ls 5 title L5, \
     ""              using "n":"accelerate" with linespoints ls 6 title L6, \
     ""              using "n":"blis_sme"  with linespoints ls 7 title L7
