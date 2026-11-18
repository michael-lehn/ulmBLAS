load "_style.gp"
set terminal svg size 1250,760 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "d_shapes_ratio.svg"
set title "dgemm, all shapes (m x n x k): relative to master\n".CPU font ",14"
set ylabel "performance relative to master"
set yrange [0.8:1.3]
set xtics rotate by -60 font ",9"
set style data histogram
set style histogram clustered gap 1
set style fill solid 0.85 border -1
set boxwidth 0.9
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
plot D."d_shapes.csv" using "generic_ratio":xticlabels(2) lc rgb "#2a78d6" title "redesign, hand-written table", \
     D."d_shapes.csv" using "tuned_ratio" lc rgb "#e34948" title "redesign, table of make tune", \
     1 with lines lc rgb "#52514e" dt 2 notitle
