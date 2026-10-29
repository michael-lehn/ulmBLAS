load "_style.gp"
set terminal svg size 1100,600 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "blocksize_lib.svg"
set title "Block sizes in the library (8fe961a with MC, KC, NC of double replaced, bench2, 3 interleaved rounds)\nGFLOP/s relative to the block sizes of BLIS PR #921 (MC = 608, KC = 2240, NC = 3072)"
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.9 border rgb "#404040"
set boxwidth 0.95
set grid noxtics
set xtics rotate by -30 font ",12"
set yrange [0.85:1.5]
set ytics 0.05
set xlabel "matrix shape m x n x k"
set ylabel "relative performance"
set key inside top left vertical maxrows 4 samplen 1.5 font ",11" nobox
plot S."blocksizes_lib.csv" using "b608_2240_3072_rel":xtic(1) lc rgb "#f0f0f0" title "608, 2240, 3072 (BLIS)", \
     "" using "b608_1024_3072_rel" lc rgb "#c6dbef" title "608, 1024, 3072", \
     "" using "b608_1024_2048_rel" lc rgb "#9ecae1" title "608, 1024, 2048", \
     "" using "b384_1536_2048_rel" lc rgb "#6baed6" title "384, 1536, 2048", \
     "" using "b384_1024_3072_rel" lc rgb "#4292c6" title "384, 1024, 3072", \
     "" using "b384_1024_2048_rel" lc rgb "#2171b5" title "384, 1024, 2048", \
     "" using "b384_1024_1024_rel" lc rgb "#08306b" title "384, 1024, 1024 (8fe961a)"
