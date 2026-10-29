load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "full_range.svg"
set title "DGEMM, square matrices n = 2 … 4096: published vs. final ulmBLAS SME2, Apple M5, one performance core\nmedians of 3 interleaved rounds; dotted: FMOPA and NEON FMA peaks (microbenchmarks)" font ",14"
set xlabel "matrix size n"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set xtics 0, 512, 4096
set mxtics 2
set yrange [0:580]
set key inside right bottom vertical font ",11.5" spacing 1.05 box opaque
plot SMEPEAK with lines ls 9 notitle, NEONPEAK with lines ls 9 notitle, \
     S."squares.csv" using "n":"published" with linespoints ls 1 title L1, \
     ""              using "n":"final"     with linespoints ls 5 title L5, \
     ""              using "n":"accelerate" with linespoints ls 6 title L6, \
     ""              using "n":"blis_sme"  with linespoints ls 7 title L7
