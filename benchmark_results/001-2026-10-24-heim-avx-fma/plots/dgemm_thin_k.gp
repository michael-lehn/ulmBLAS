load "_style.gp"
set output "dgemm_thin_k.svg"
set title "DGEMM with small k (rank-k updates), alpha = 1, beta = 1\nIntel Core i5-12500 (heim), one core"
set xlabel "matrix shape m x n x k"
set ylabel "performance [GFLOP/s]"
set grid noxtics
set style data histograms
set style histogram clustered gap 1.5
set style fill solid 0.9 border rgb "#404040"
set boxwidth 0.95
set xtics rotate by -15 offset -1,0
set bmargin 5
set key inside top left vertical maxrows 3 font ",12" samplen 1.5
set yrange [0:80]
plot S."dgemm_thin_k.csv" using "sse_mc384":xtic(1) ls 2 lw 1 title "ulmBLAS SSE", \
     ""                   using "avx_mc72"           ls 3 lw 1 title "ulmBLAS AVX/FMA (MC=72, run A)", \
     ""                   using "avx_mc288"          ls 4 lw 1 title "ulmBLAS AVX/FMA (MC=288)", \
     ""                   using "blis"               ls 5 lw 1 title "BLIS 2.0", \
     ""                   using "openblas"           ls 6 lw 1 title "OpenBLAS 0.3.34"
