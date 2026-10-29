load "_style.gp"
set terminal svg size 1250,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "ratio_tuning.svg"
set title "The three tuning iterations of the OpenBLAS kernel inside ulmBLAS, relative to native OpenBLAS 0.3.34\n".CPU.", DGEMM m = n = k, single-threaded\nmedians of the per-round ratios; band (iteration 3): minimum to maximum over the rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance relative to native OpenBLAS"
set xrange [0:4050]
set xtics 500
set yrange [0.80:1.10]
set ytics 0.05
set mytics 5
set grid mytics lc rgb "#ededed"
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
set arrow from 0,1 to 4050,1 nohead lc rgb "#0b0b0b" lw 1.2
set label "values below 0.80 (smallest sizes) are cut off here; full range: ratio_openblas.svg" at graph 0.98, graph 0.04 right font ",11" noenhanced textcolor rgb "#52514e"
I = "ulmblas_openblas_skylakex_16x12_it"
plot D."ratios.csv" using "n":(column(I."3_over_openblas_min")):(column(I."3_over_openblas_max")) \
         with filledcurves fs transparent solid 0.18 noborder lc rgb C[1] notitle, \
     D."ratios.csv" using "n":I."1_over_openblas" with linespoints lc rgb C[1] lw 1.6 dt 3 pt 6 pi 8 ps 0.75 \
         title "iteration 1: one kernel call per 16x12 tile", \
     D."ratios.csv" using "n":I."2_over_openblas" with linespoints lc rgb C[1] lw 1.6 dt 2 pt 4 pi 8 ps 0.75 \
         title "iteration 2: one call per panel of B", \
     D."ratios.csv" using "n":I."3_over_openblas" with linespoints lc rgb C[1] lw 2.4 dt 1 pt 7 pi 8 ps 0.75 \
         title "iteration 3 (final): B packed interleaved"
