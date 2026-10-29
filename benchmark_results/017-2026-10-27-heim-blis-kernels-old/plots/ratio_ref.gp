load "_style.gp"
load "_labels.gp"
set terminal svg size 1250,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "ratio_ref.svg"
set title "Micro kernels in ulmBLAS relative to their native reference (".CPU.")\nratio of the medians; dotted: 0.95, 0.90, 0.85" font ",14"
set xlabel "matrix size n"
set ylabel "ulmBLAS / native"
set logscale x 2
set xrange [6:4800]
set yrange [0.5:1.5]
set key outside right center vertical font ",10.5" box opaque
plot for [i=1:|R|] D."dgemm_square.csv" using "n":R[i] with linespoints lc rgb color(i) lw 2 pt 7 ps 0.5 title RT[i], \
     0.95 with lines lc rgb "#808080" dt 3 notitle, 0.90 with lines lc rgb "#808080" dt 3 notitle, \
     0.85 with lines lc rgb "#808080" dt 3 notitle, 1 with lines lc rgb "#404040" lw 1 notitle
