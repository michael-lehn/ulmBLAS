load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "small_time.svg"
set title "DGEMM, tiny square matrices: time per call (lower is better), Apple M5, one performance core\nmedians of 3 interleaved rounds; BLIS PR #921 SME2 (1.8 to 4 µs) is above the range" font ",14"
set xlabel "matrix size n"
set ylabel "time per call [µs]"
set xrange [1:33]
set xtics 0, 4, 32
set mxtics 4
set yrange [0:2.0]
set ytics 0, 0.2
set key inside left top vertical font ",11.5" spacing 1.05 box opaque
plot S."squares.csv" using "n":"published_us" with linespoints ls 1 title L1, \
     ""              using "n":"packtail_us"  with linespoints ls 2 title L2, \
     ""              using "n":"cache_us"     with linespoints ls 3 title L3, \
     ""              using "n":"direct_us"    with linespoints ls 4 title L4, \
     ""              using "n":"final_us"     with linespoints ls 5 title L5, \
     ""              using "n":"accelerate_us" with linespoints ls 6 title L6
