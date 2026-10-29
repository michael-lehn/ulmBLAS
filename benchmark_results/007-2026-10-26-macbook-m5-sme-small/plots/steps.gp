load "_style.gp"
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "steps.svg"
set title "Speedup of each ulmBLAS version over the published one, square matrices n = 2 … 128\n(time of published / time of the version; medians of 3 interleaved rounds)" font ",14"
set xlabel "matrix size n"
set ylabel "speedup over published"
set xrange [0:130]
set xtics 0, 16, 128
set mxtics 2
set logscale y 2
set yrange [0.8:64]
set ytics (1, 2, 4, 8, 16, 32, 64)
set key inside right top vertical font ",11.5" spacing 1.05 box opaque
plot 1 with lines ls 9 notitle, \
     S."squares.csv" using "n":(column("packtail")/column("published")) with linespoints ls 2 title L2, \
     ""              using "n":(column("cache")/column("published"))    with linespoints ls 3 title L3, \
     ""              using "n":(column("direct")/column("published"))   with linespoints ls 4 title L4, \
     ""              using "n":(column("final")/column("published"))    with linespoints ls 5 title L5, \
     ""              using "n":(column("accelerate")/column("published")) with linespoints ls 6 title "Apple Accelerate (for comparison)"
