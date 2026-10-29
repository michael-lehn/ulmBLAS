load "_style.gp"
set terminal svg size 1250,620 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "ratio_native.svg"
set title "ulmBLAS relative to the native library with the same kernel, SGEMM m = n = k, single-threaded\n".CPU."\nmedians of the per-round ratios; bands: minimum to maximum over the rounds" font ",14"
set xlabel "matrix size n"
set ylabel "performance relative to the native library"
set xrange [0:4050]
set xtics 500
set yrange [0.4:1.4]
set ytics 0.1
set mytics 2
set grid mytics lc rgb "#ededed"
set key outside right center vertical font ",11.5" spacing 1.2 width 1 box lc rgb "#dcdcdc"
set arrow from 0,1 to 4050,1 nohead lc rgb "#0b0b0b" lw 1.2
array I[4] = [1, 2, 4, 5]
array REF[4] = ["openblas", "blis_auto", "blis_haswell", "blis_haswell"]
array RT[4] = [" / OpenBLAS", " / BLIS skx", " / BLIS haswell", " / BLIS haswell"]
plot for [j=1:4] D."ratios.csv" using "n":(column(V[I[j]]."_over_".REF[j]."_min")):(column(V[I[j]]."_over_".REF[j]."_max")) \
         with filledcurves fs transparent solid 0.18 noborder lc rgb C[I[j]] notitle, \
     for [j=1:4] D."ratios.csv" using "n":V[I[j]]."_over_".REF[j] with linespoints \
         lc rgb C[I[j]] lw 2.2 pt P[I[j]] pi 8 ps 0.75 title T[I[j]].RT[j]
