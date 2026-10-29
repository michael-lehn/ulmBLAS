#  Where the time goes: the OpenBLAS kernel inside ulmBLAS, step by step from
#  the library to the kernel alone, and native OpenBLAS (data/summary/stages.csv)
set datafile separator ","
set datafile columnheaders
set terminal svg size 1150,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "stages.svg"
set title "OpenBLAS SKYLAKEX kernel inside ulmBLAS (MC/KC/NC 192/384/8640), Intel Xeon Platinum 8358, one core\nmedians of 3 runs; note: y axis from 80" font ",14"
set ylabel "performance [GFLOP/s]"
set yrange [80:104]
set grid ytics lc rgb "#d0d0d0"
set style data histograms
set style histogram clustered gap 1.5
set style fill solid 0.9 noborder
set boxwidth 0.9
set key outside right center vertical font ",11" box opaque
set xlabel "m = n = k"
plot "../data/summary/stages.csv" using "lib":xtic(1) lc rgb "#2a78d6" title "ulmBLAS (library)", \
     "" using "macro"       lc rgb "#8fb9ea" title "kernel called per block, with packing", \
     "" using "nopack"      lc rgb "#c8323c" title "macro kernel, no packing", \
     "" using "macronopack" lc rgb "#e8a0a5" title "kernel called per block, no packing", \
     "" using "kern"        lc rgb "#3a3a38" title "micro kernel alone (L1/L2 panels)", \
     "" using "native_openblas" lc rgb "#1baf7a" title "native OpenBLAS 0.3.34"
