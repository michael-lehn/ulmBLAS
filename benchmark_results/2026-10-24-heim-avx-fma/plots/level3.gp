load "_style.gp"
set terminal svg size 1100,820 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "level3.svg"
set xrange [400:3300]
set yrange [0:70]
set logscale x 2
set xtics (512, 1024, 2048, 3072)
set multiplot layout 2,2 title "Level 3 routines (n x n, column major, lower, left), Intel Core i5-12500 (heim), one core" font ",16"
do for [r in "dsyrk dsymm dtrmm dtrsm"] {
    if (r eq "dsymm") {
        set key inside center center vertical font ",11" spacing 1.1 nobox
    } else {
        set key off
    }
    set title sprintf("%s", r) font ",15"
    set xlabel "n"
    set ylabel "performance [GFLOP/s]"
    plot S."level3_".r.".csv" using "n":"openblas"      with linespoints ls 6 title "OpenBLAS 0.3.34", \
         ""                   using "n":"blis"          with linespoints ls 5 title "BLIS 2.0", \
         ""                   using "n":"avx_mc288"     with linespoints ls 4 title "ulmBLAS AVX/FMA (MC=288)", \
         ""                   using "n":"runA_avx_mc72" with linespoints ls 3 title "ulmBLAS AVX/FMA (MC=72, run A)", \
         ""                   using "n":"sse_mc384"     with linespoints ls 2 title "ulmBLAS SSE"
}
unset multiplot
