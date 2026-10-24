load "_style.gp"
set terminal svg size 1100,520 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "blocksize.svg"
set multiplot layout 1,2 title "Choice of MC and KC for the NEON micro kernel (interleaved runs A2, A3), Apple M5 (MacBook Air), one performance core" font ",15"
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.9 border rgb "#404040"
set boxwidth 0.95
set grid noxtics
set xtics rotate by -20
set ytics 0.01
set yrange [0.94:1.05]
set key inside top left horizontal maxrows 3 samplen 1.5 font ",11"
set title "DGEMM, beta = 1 (run A2)"
set xlabel "matrix shape m x n x k"
set ylabel "GFLOP/s relative to MC=240, KC=256"
plot S."blocksize_dgemm.csv" using "mc120_kc1024_rel":xtic(1) lc rgb "#c6dbef" title "MC=120, KC=1024", \
     "" using "mc240_kc1024_rel" lc rgb "#9ecae1" title "MC=240, KC=1024", \
     "" using "mc480_kc1024_rel" lc rgb "#4292c6" title "MC=480, KC=1024", \
     "" using "mc960_kc1024_rel" lc rgb "#08519c" title "MC=960, KC=1024"
set title "syrk, symm, trmm, trsm (run A3)"
set xlabel ""
set key inside top center horizontal maxrows 1 samplen 1.5 font ",10" width -1
set xtics rotate by -35 font ",11"
set yrange [0.94:1.07]
plot S."blocksize_level3.csv" using "mc240_kc1024_rel":xtic(1) lc rgb "#9ecae1" title "MC=240, KC=1024", \
     "" using "mc480_kc1024_rel" lc rgb "#4292c6" title "MC=480, KC=1024", \
     "" using "mc960_kc1024_rel" lc rgb "#08519c" title "MC=960, KC=1024"
unset multiplot
