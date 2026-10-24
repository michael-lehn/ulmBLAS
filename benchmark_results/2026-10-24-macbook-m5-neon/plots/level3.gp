load "_style.gp"
set terminal svg size 1100,820 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "level3.svg"
set xrange [400:3300]
set yrange [0:75]
set logscale x 2
set xtics (512, 1024, 2048, 3072)
set multiplot layout 2,2 title "Level 3 routines (n x n, column major, lower, left), Apple M5 (MacBook Air), one performance core" font ",16"
do for [r in "dsyrk dsymm dtrmm dtrsm"] {
    if (r eq "dsymm") {
        set key inside center bottom vertical font ",11" spacing 1.1 nobox
    } else {
        set key off
    }
    set title sprintf("%s", r) font ",15"
    set xlabel "n"
    set ylabel "performance [GFLOP/s]"
    plot S."level3_".r.".csv" using "n":"openblas" with linespoints ls 6 title "OpenBLAS 0.3.34", \
         ""                   using "n":"blis"     with linespoints ls 5 title "BLIS 2.0", \
         ""                   using "n":"neon"     with linespoints ls 4 title "ulmBLAS NEON", \
         ""                   using "n":"base"     with linespoints ls 1 title "ulmBLAS before (reference kernel)"
}
unset multiplot
