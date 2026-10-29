load "_style.gp"
set terminal svg size 1100,820 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "level3.svg"
set xrange [0:2200]
set yrange [0:580]
set xtics 0, 512, 2048
set mxtics 2
set multiplot layout 2,2 title "Level 3 routines (n x n, column major, lower, left), Apple M5 (MacBook Air), one performance core" font ",16"
do for [r in "dsyrk dsymm dtrmm dtrsm"] {
    if (r eq "dtrsm") {
        set key inside left top vertical font ",11" spacing 1.1 nobox
    } else {
        set key off
    }
    set title sprintf("%s", r) font ",15"
    set xlabel "n"
    set ylabel "performance [GFLOP/s]"
    plot S."level3_".r.".csv" using "case":"sme"        with linespoints ls 1 title "ulmBLAS SME2", \
         ""                   using "case":"blis_sme"   with linespoints ls 2 title "BLIS PR #921 SME2", \
         ""                   using "case":"accelerate" with linespoints ls 3 title "Apple Accelerate", \
         ""                   using "case":"neon"       with linespoints ls 4 title "ulmBLAS NEON", \
         ""                   using "case":"blis"       with linespoints ls 5 title "BLIS 2.0 NEON", \
         ""                   using "case":"openblas"   with linespoints ls 6 title "OpenBLAS 0.3.34"
}
unset multiplot
