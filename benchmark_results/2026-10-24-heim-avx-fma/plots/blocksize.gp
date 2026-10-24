load "_style.gp"
set terminal svg size 1100,520 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "blocksize.svg"
set multiplot layout 1,2 title "Choice of MC and KC for the AVX/FMA micro kernel (run A), Intel Core i5-12500 (heim), one core" font ",15"
set title "DGEMM, beta = 1: MC, KC sweep (NC = 4080)"
set xlabel "matrix shape m x n x k"
set ylabel "performance [GFLOP/s]"
set yrange [55:61]
set grid noxtics
set key inside top left horizontal maxrows 3 samplen 1.5 font ",11"
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.9 border rgb "#404040"
set boxwidth 0.95
set xtics rotate by -20
plot S."dgemm_sweep.csv" using "mc72_kc256":xtic(1) lc rgb "#c6dbef" title "MC=72, KC=256", \
     "" using "mc72_kc384"  lc rgb "#9ecae1" title "MC=72, KC=384", \
     "" using "mc72_kc512"  lc rgb "#6baed6" title "MC=72, KC=512", \
     "" using "mc144_kc256" lc rgb "#c7e9c0" title "MC=144, KC=256", \
     "" using "mc144_kc384" lc rgb "#a1d99b" title "MC=144, KC=384", \
     "" using "mc144_kc512" lc rgb "#74c476" title "MC=144, KC=512", \
     "" using "mc288_kc256" lc rgb "#08519c" title "MC=288, KC=256", \
     "" using "mc288_kc384" lc rgb "#2171b5" title "MC=288, KC=384", \
     "" using "mc288_kc512" lc rgb "#4292c6" title "MC=288, KC=512"
set title "syrk, symm, trmm, trsm: MC = 288 relative to MC = 72 (KC = 256)"
set style data linespoints
set xtics norotate
set xlabel "n"
set ylabel "GFLOP/s (MC=288) / GFLOP/s (MC=72)"
set yrange [0.95:1.15]
set ytics 0.05
set logscale x 2
set xrange [400:3300]
set xtics (512, 1024, 2048, 3072)
set grid xtics ytics
set key inside top left vertical maxrows 4 font ",12"
plot 1 with lines ls 9 notitle, \
     S."level3_dsyrk.csv" using "n":"runA_mc288_over_mc72" lw 2.2 pt 5 lc rgb "#08519c" title "dsyrk", \
     S."level3_dsymm.csv" using "n":"runA_mc288_over_mc72" lw 2.2 pt 7 lc rgb "#238b45" title "dsymm", \
     S."level3_dtrmm.csv" using "n":"runA_mc288_over_mc72" lw 2.2 pt 9 lc rgb "#e08214" title "dtrmm", \
     S."level3_dtrsm.csv" using "n":"runA_mc288_over_mc72" lw 2.2 pt 13 lc rgb "#cb181d" title "dtrsm"
unset multiplot
