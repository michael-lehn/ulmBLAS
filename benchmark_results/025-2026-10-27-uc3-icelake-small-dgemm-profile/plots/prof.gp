load "_style.gp"
set terminal svg size 1150,560 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "prof.svg"
set multiplot layout 1,2 title "Where ulmBLAS DGEMM spends its time for small n (Ice Lake, prof-ob of campaign 016, SKYLAKEX kernel, medians of 3 rounds)" font ",13"
F = D."prof.csv"
m(mode, c) = (strcol("mode") eq mode) ? column(c) : NaN
set xlabel "matrix size n (m = n = k)"
set xrange [0:250]
set ylabel "fraction of the time of a call (instrumented driver)"
set yrange [0:1]
set key top right font ",11"
plot F using "n":(m("inst","packB")+m("inst","packA")) with linespoints lc rgb "#e8a33d" lw 2 pt 7 title "packing A and B", \
     F using "n":(m("inst","macrokernel")) with linespoints lc rgb "#2a78d6" lw 2 pt 5 title "macro kernel", \
     F using "n":(m("inst","other")) with linespoints lc rgb "#6b6a64" lw 1.6 pt 9 title "rest (block loops, pool)"
set ylabel "performance [GFLOP/s]"
set yrange [0:105]
set key bottom right font ",11"
plot F using "n":(m("kern","gflops")) with linespoints lc rgb "#1baf7a" lw 1.6 pt 13 title "micro kernel, full tiles in cache", \
     F using "n":(m("nopack","gflops")) with linespoints lc rgb "#7d4fc4" lw 1.6 pt 9 title "macro kernel, no packing", \
     F using "n":(m("lib","gflops")) with linespoints ls 1 title "ulmBLAS DGEMM", \
     F using "n":(m("cblas","gflops")) with linespoints ls 3 title "OpenBLAS DGEMM"
unset multiplot
