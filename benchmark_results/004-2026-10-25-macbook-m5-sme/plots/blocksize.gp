load "_style.gp"
set terminal svg size 1100,560 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "blocksize.svg"
set title "Block sizes of the SME2 configuration (prof -m inst, kernel and packing of 889aa15, data/raw/blocks/sweep.txt)\nGFLOP/s relative to the block sizes of BLIS PR #921 (MC = 608, KC = 2240, NC = 3072)"
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.9 border rgb "#404040"
set boxwidth 0.95
set grid noxtics
set yrange [0.9:1.32]
set ytics 0.05
set xlabel "matrix shape m x n x k"
set ylabel "relative performance"
set key inside top left vertical maxrows 4 samplen 1.5 font ",11" nobox
plot S."blocksizes.csv" using "mc608_kc2240_nc3072":xtic(1) lc rgb "#f0f0f0" title "MC=608, KC=2240, NC=3072 (BLIS)", \
     "" using "mc384_kc2240_nc3072" lc rgb "#c6dbef" title "MC=384, KC=2240, NC=3072", \
     "" using "mc384_kc1024_nc3072" lc rgb "#9ecae1" title "MC=384, KC=1024, NC=3072", \
     "" using "mc608_kc1024_nc1024" lc rgb "#6baed6" title "MC=608, KC=1024, NC=1024", \
     "" using "mc256_kc1024_nc1024" lc rgb "#4292c6" title "MC=256, KC=1024, NC=1024", \
     "" using "mc384_kc1024_nc1024" lc rgb "#08519c" title "MC=384, KC=1024, NC=1024 (chosen)", \
     "" using "mc384_kc1536_nc1024" lc rgb "#08306b" title "MC=384, KC=1536, NC=1024"
