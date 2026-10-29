load "_style.gp"
set output "packing.svg"
set title "Packing variants (driver of prof.cc, block sizes of ulmBLAS, run P3)\nApple M5 (MacBook Air), one performance core"
set style histogram clustered gap 1
set key inside top left horizontal maxrows 2 samplen 1.5 font ",11"
set xlabel "matrix shape m x n x k"
set ylabel "GFLOP/s relative to the packing of d25e3eb"
set yrange [0.96:1.12]
set ytics 0.02
plot S."packing.csv" using "unit:unit_rel":xtic(1) lc rgb "#c6dbef" title "A, B: unit stride loops", \
     "" using "blk4:unit_rel" lc rgb "#6baed6" title "A: 4 panels together", \
     "" using "blk8:unit_rel" lc rgb "#2171b5" title "A: 8 panels together", \
     "" using "blk4:trn_rel"  lc rgb "#a1d99b" title "A: 4 panels, B: NEON 2x2 transposes", \
     "" using "blk8:trn_rel"  lc rgb "#238b45" title "A: 8 panels, B: NEON 2x2 transposes", \
     1 with lines ls 9 notitle
